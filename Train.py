import csv
import json
import sys
import time

import numpy as np
from pyspark import SparkConf, SparkContext
from xgboost import XGBRegressor

def save_predictions(predictions, filename):
    headers = ["user_id", "business_id", "prediction"]
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(predictions)

class CollaborativeFiltering:
    def __init__(self):
        pass

    @staticmethod
    def preprocess_rdd(rdd, mode="train"):
        header = rdd.first()
        rdd = rdd.filter(lambda line: line != header).map(lambda line: line.split(","))
        if mode == "train":
            return rdd.map(lambda fields: (fields[0], fields[1], float(fields[2])))
        return rdd.map(lambda fields: (fields[0], fields[1]))

    @staticmethod
    def build_item_user_map(train_rdd):
        return (
            train_rdd.map(lambda x: (x[1], (x[0], x[2])))
            .groupByKey()
            .mapValues(lambda ratings: {
                "users": dict(ratings),
                "avg_rating": np.mean([rating for _, rating in ratings])
            })
            .collectAsMap()
        )

    @staticmethod
    def build_user_item_map(train_rdd):
        return (
            train_rdd.map(lambda x: (x[0], (x[1], x[2])))
            .groupByKey()
            .mapValues(lambda items: {"items": dict(items)})
            .collectAsMap()
        )

    @staticmethod
    def pearson_similarity(pair, item_user_map):
        item1, item2 = pair
        users_item1 = set(item_user_map[item1]["users"].keys())
        users_item2 = set(item_user_map[item2]["users"].keys())
        common_users = users_item1.intersection(users_item2)

        if len(common_users) <= 1:
            return (5 - abs(item_user_map[item1]["avg_rating"] - item_user_map[item2]["avg_rating"])) / 5

        ratings1 = np.array([item_user_map[item1]["users"][user] for user in common_users])
        ratings2 = np.array([item_user_map[item2]["users"][user] for user in common_users])
        mean1, mean2 = ratings1.mean(), ratings2.mean()
        ratings1 -= mean1
        ratings2 -= mean2

        numerator = np.sum(ratings1 * ratings2)
        denominator = np.sqrt(np.sum(ratings1**2)) * np.sqrt(np.sum(ratings2**2))

        return 0 if denominator == 0 else numerator / denominator

    @staticmethod
    def predict_rating(user_business_pair, item_user_map, user_item_map, num_neighbors=15):
        user, business = user_business_pair
        if user not in user_item_map or business not in item_user_map:
            return 3.0

        similarities = [
            (CollaborativeFiltering.pearson_similarity((business, item), item_user_map), item_user_map[item]["users"][user])
            for item in user_item_map[user]["items"].keys()
        ]

        top_similarities = sorted(similarities, key=lambda x: -x[0])[:num_neighbors]
        weighted_sum = np.sum([sim * rating for sim, rating in top_similarities])
        total_weight = np.sum([abs(sim) for sim, _ in top_similarities])

        return 3.5 if total_weight == 0 else weighted_sum / total_weight

    def execute(self, spark, train_file, test_file):
        train_rdd = spark.textFile(train_file)
        train_rdd = CollaborativeFiltering.preprocess_rdd(train_rdd, mode="train")

        item_user_map = CollaborativeFiltering.build_item_user_map(train_rdd)
        user_item_map = CollaborativeFiltering.build_user_item_map(train_rdd)

        test_rdd = spark.textFile(test_file)
        test_rdd = CollaborativeFiltering.preprocess_rdd(test_rdd, mode="valid").cache()

        predictions = test_rdd.map(
            lambda fields: [fields[0], fields[1], CollaborativeFiltering.predict_rating((fields[0], fields[1]), item_user_map, user_item_map)]
        ).cache()

        return predictions


class ModelBasedApproach:
    def __init__(self):
        pass

    @staticmethod
    def load_csv(path, sc):
        rdd = sc.textFile(path)
        header = rdd.first()
        return rdd.filter(lambda line: line != header).map(lambda line: line.split(","))

    @staticmethod
    def load_json(path, sc):
        return sc.textFile(path).map(lambda line: json.loads(line))

    @staticmethod
    def process_reviews(review_rdd):
        review_map = (
            review_rdd.map(
                lambda record: (record["business_id"], (float(record["useful"]), float(record["funny"]), float(record["cool"])))
            )
            .groupByKey()
            .mapValues(lambda ratings: np.mean(np.array(ratings), axis=0))
            .collectAsMap()
        )
        return review_map

    @staticmethod
    def process_users(user_rdd):
        return user_rdd.map(
            lambda record: (record["user_id"], (float(record["average_stars"]), float(record["review_count"]), float(record["fans"])))
        ).collectAsMap()

    @staticmethod
    def process_businesses(bus_rdd):
        return bus_rdd.map(
            lambda record: (record["business_id"], (float(record["stars"]), float(record["review_count"])))
        ).collectAsMap()

    @staticmethod
    def transform_row(row, review_map, user_map, bus_map):
        if len(row) == 3:
            user, bus, rating = row
        else:
            user, bus = row
            rating = None

        review_data = review_map.get(bus, (None, None, None))
        user_data = user_map.get(user, (None, None, None))
        bus_data = bus_map.get(bus, (None, None))

        return (review_data + user_data + bus_data, rating)

    def execute(self, spark, folder_path, test_file):
        train_rdd = ModelBasedApproach.load_csv(f"{folder_path}/yelp_train.csv", spark)
        review_rdd = ModelBasedApproach.load_json(f"{folder_path}/review_train.json", spark)
        review_map = ModelBasedApproach.process_reviews(review_rdd)

        user_rdd = ModelBasedApproach.load_json(f"{folder_path}/user.json", spark)
        user_map = ModelBasedApproach.process_users(user_rdd)

        bus_rdd = ModelBasedApproach.load_json(f"{folder_path}/business.json", spark)
        bus_map = ModelBasedApproach.process_businesses(bus_rdd)

        val_rdd = ModelBasedApproach.load_csv(test_file, spark).cache()
        train_rdd = train_rdd.map(lambda row: ModelBasedApproach.transform_row(row, review_map, user_map, bus_map))
        val_processed = val_rdd.map(lambda row: ModelBasedApproach.transform_row(row, review_map, user_map, bus_map))

        X_train = np.array(train_rdd.map(lambda x: x[0]).collect(), dtype="float32")
        Y_train = np.array(train_rdd.map(lambda x: x[1]).collect(), dtype="float32")
        X_val = np.array(val_processed.map(lambda x: x[0]).collect(), dtype="float32")

        model = XGBRegressor()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_val)

        result = [[row[0], row[1], pred] for row, pred in zip(val_rdd.collect(), predictions)]
        return spark.parallelize(result)


def combine_predictions(predictions, weight=0.5):
    return weight * predictions[0] + (1 - weight) * predictions[1]


def main(data_folder, test_file, output_file):
    conf = SparkConf().setAppName("Hybrid Recommendation System")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    try:
        start_time = time.time()

        cf = CollaborativeFiltering()
        cf_predictions = cf.execute(sc, f"{data_folder}/yelp_train.csv", test_file)
        cf_predictions = cf_predictions.map(lambda x: ((x[0], x[1]), x[2])).persist()

        mba = ModelBasedApproach()
        mba_predictions = mba.execute(sc, data_folder, test_file)
        mba_predictions = mba_predictions.map(lambda x: ((x[0], x[1]), x[2])).persist()

        FACTOR = 0.05222

        combined_preds = (
            cf_predictions.join(mba_predictions)
            .map(lambda x: [x[0][0], x[0][1], combine_predictions(x[1], weight=FACTOR)])
