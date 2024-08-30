import xgboost as xgb
from xgboost import XGBRegressor
import json
import csv
import time
import math
from datetime import datetime
from pyspark import SparkContext, SparkConf
import sys
import os

# Initialize Spark context
def initialize_spark():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    sc = SparkContext()
    sc.setLogLevel('WARN')
    return sc

# Predict ratings function
def predict_rating(row, business_map, user_map, user_avg_ratings, business_avg_ratings):
    business, user = row
    if business not in business_map and user not in user_map:
        return user, business, 2.5, 1
    if business not in business_map:
        return user, business, user_avg_ratings.get(user, 2.5), 1
    if user not in user_map:
        return user, business, business_avg_ratings.get(business, 2.5), 1

    coeffs = []
    user_ratings = business_map[business]
    business_avg = business_avg_ratings[business]
    user_data = user_map[user]

    for other_business in user_data:
        if other_business not in business_map:
            continue
        other_avg = business_avg_ratings[other_business]
        other_ratings = business_map[other_business]

        common_users = set(user_ratings.keys()) & set(other_ratings.keys())
        if len(common_users) < 23:
            diff = user_data[other_business] - other_avg
            similarity = business_avg / business_avg_ratings.get(other_business, 1)
            coeffs.append((diff, similarity))
        else:
            num = norm1 = norm2 = 0
            for user in common_users:
                r1 = user_ratings[user]
                r2 = other_ratings[user]
                norm1 += (r1 - business_avg) ** 2
                norm2 += (r2 - other_avg) ** 2
                num += (r1 - business_avg) * (r2 - other_avg)

            denom = math.sqrt(norm1 * norm2)
            similarity = num / denom if denom != 0 else 0
            coeffs.append((other_ratings[user] - other_avg, similarity))

    coeffs = sorted(coeffs, key=lambda x: abs(x[1]), reverse=True)[:65536]
    numerator = sum(c * r for r, c in coeffs)
    denominator = sum(abs(c) for _, c in coeffs)
    alpha = 1 * len(coeffs) / len(business_avg_ratings)
    predicted_rating = business_avg + (numerator / denominator if denominator != 0 else 0)

    return user, business, max(1, min(predicted_rating, 5)), alpha

# Item-based recommendation system
def item_based(train_path, test_path, sc):
    data_rdd = sc.textFile(train_path)
    header = data_rdd.first()
    data_rdd = data_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
    data_rdd = data_rdd.map(lambda x: (x[1], (x[0], float(x[2]))))

    business_ratings = data_rdd.groupByKey().mapValues(lambda x: dict(x)).filter(lambda x: len(x[1]) >= 23)
    business_map = business_ratings.collectAsMap()

    user_ratings = data_rdd.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().mapValues(lambda x: dict(x))
    user_map = user_ratings.collectAsMap()
    user_avg_ratings = user_ratings.mapValues(lambda x: sum(x.values()) / len(x)).collectAsMap()

    business_avg_ratings = business_ratings.mapValues(lambda x: sum(x.values()) / len(x)).collectAsMap()

    test_rdd = sc.textFile(test_path)
    header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda x: x != header).map(lambda x: x.split(",")[:2])
    results = test_rdd.map(lambda x: predict_rating(x, business_map, user_map, user_avg_ratings, business_avg_ratings))
    return results.toLocalIterator()

# Function to process features and ratings for model-based recommendations
def process_features_and_ratings(folder_path, sc):
    def extract_features(record, keys):
        features = []
        for key in keys:
            parts = key.split(".")
            value = record
            for part in parts[:-1]:
                value = value.get(part, {})
            if value is not None:
                value = value.get(parts[-1], None)
                if parts[-1] == 'yelping_since':
                    value = (datetime.now() - datetime.strptime(value, "%Y-%m-%d")).total_seconds()
                elif isinstance(value, str):
                    value = 1 if value == "True" else 0 if value == "False" else value
            features.append(value)
        return features

    rating_rdd = sc.textFile(f"{folder_path}/yelp_train.csv").filter(lambda x: x != 'user_id,business_id,stars').map(lambda x: x.split(","))
    user_rdd = sc.textFile(f"{folder_path}/user.json").map(lambda x: json.loads(x)).map(lambda x: extract_features(x, user_keys))
    business_rdd = sc.textFile(f"{folder_path}/business.json").map(lambda x: json.loads(x)).map(lambda x: extract_features(x, business_keys))

    user_map = user_rdd.aggregateByKey([], lambda acc, x: acc + [x], lambda acc1, acc2: acc1 + acc2).mapValues(lambda x: [sum(col) / len(col) for col in zip(*x)]).collectAsMap()
    business_map = business_rdd.aggregateByKey([], lambda acc, x: acc + [x], lambda acc1, acc2: acc1 + acc2).mapValues(lambda x: [sum(col) / len(col) for col in zip(*x)]).collectAsMap()

    tip_rdd = sc.textFile(f"{folder_path}/tip.json").map(lambda x: json.loads(x))
    user_tip_count = tip_rdd.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
    business_tip_count = tip_rdd.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()

    mean_user_tips = sum(user_tip_count.values()) / len(user_tip_count)
    mean_business_tips = sum(business_tip_count.values()) / len(business_tip_count)

    photo_rdd = sc.textFile(f"{folder_path}/photo.json").map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
    mean_photos = sum(photo_rdd.values()) / len(photo_rdd)

    return user_map, business_map, user_tip_count, business_tip_count, mean_user_tips, mean_business_tips, photo_rdd, mean_photos

# Main function for model-based recommendations
def model_based(folder_path, test_path, sc):
    user_map, business_map, user_tip_count, business_tip_count, mean_user_tips, mean_business_tips, photo_rdd, mean_photos = process_features_and_ratings(folder_path, sc)

    rating_rdd = sc.textFile(f"{folder_path}/yelp_train.csv").filter(lambda x: x != 'user_id,business_id,stars').map(lambda x: x.split(","))
    train_features_rdd = rating_rdd.map(lambda x: (x[0], x[1:])).join(user_rdd).join(business_rdd).mapValues(lambda x: x[0] + x[1] + [user_tip_count.get(x[0], mean_user_tips), business_tip_count.get(x[1], mean_business_tips), photo_rdd.get(x[1], mean_photos)])
    train_features = train_features_rdd.map(lambda x: x[1][:-1])
    train_ratings = train_features_rdd.map(lambda x: x[1][-1])

    best_params = {
        'max_depth': 5, 'learning_rate': 0.07, 'subsample': 0.9, 'colsample_bytree': 0.8, 
        'colsample_bylevel': 0.7, 'n_estimators': 1000, 'min_child_weight': 5, 'tree_method': 'hist', 
        'gamma': 0.19, 'random_state': 51
    }

    xgb_model = XGBRegressor(**best_params, verbosity=0)
    xgb_model.fit(train_features.collect(), train_ratings.collect())

    test_rdd = sc.textFile(test_path).filter(lambda x: x != 'user_id,business_id').map(lambda x: x.split(",")[:2])
    test_features_rdd = test_rdd.map(lambda x: (x, (user_map.get(x[0], [0]*len(user_keys)), business_map.get(x[1], [0]*len(business_keys
