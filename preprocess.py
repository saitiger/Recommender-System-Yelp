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

def load_csv(path, sc):
    rdd = sc.textFile(path)
    header = rdd.first()
    return rdd.filter(lambda line: line != header).map(lambda line: line.split(","))

def load_json(path, sc):
    return sc.textFile(path).map(lambda line: json.loads(line))

def compute_review_averages(review_rdd):
    processed_reviews = (
        review_rdd.map(lambda record: (record["business_id"], (float(record["useful"]), float(record["funny"]), float(record["cool"]))))
        .groupByKey()
        .mapValues(lambda ratings: tuple(np.mean(np.array(ratings), axis=0)))
    )
    return processed_reviews.collectAsMap()

def compute_user_statistics(user_rdd):
    return user_rdd.map(
        lambda record: (record["user_id"], (float(record["average_stars"]), float(record["review_count"]), float(record["fans"])))
    ).collectAsMap()

def compute_business_statistics(bus_rdd):
    return bus_rdd.map(
        lambda record: (record["business_id"], (float(record["stars"]), float(record["review_count"])))
    ).collectAsMap()

def transform_train_data(row, review_stats, user_stats, bus_stats):
    user, bus = row[0], row[1]
    rating = float(row[2]) if len(row) == 3 else None

    review_features = review_stats.get(bus, (None, None, None))
    user_features = user_stats.get(user, (None, None, None))
    bus_features = bus_stats.get(bus, (None, None))

    return (review_features + user_features + bus_features, rating)

def run_task(folder_path, test_file, output_file):
    conf = SparkConf().setAppName("Model-based Recommendation System")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    try:
        start_time = time.time()

        train_data_rdd = load_csv(folder_path + "/yelp_train.csv", sc)
        review_data_rdd = load_json(folder_path + "/review_train.json", sc)
        user_data_rdd = load_json(folder_path + "/user.json", sc)
        business_data_rdd = load_json(folder_path + "/business.json", sc)

        review_stats = compute_review_averages(review_data_rdd)
        user_stats = compute_user_statistics(user_data_rdd)
        bus_stats = compute_business_statistics(business_data_rdd)

        train_data_rdd = train_data_rdd.map(lambda row: transform_train_data(row, review_stats, user_stats, bus_stats))
        validation_data_rdd = load_csv(test_file, sc).map(lambda row: transform_train_data(row, review_stats, user_stats, bus_stats)).cache()

        X_train = np.array(train_data_rdd.map(lambda x: x[0]).collect(), dtype="float32")
        Y_train = np.array(train_data_rdd.map(lambda x: x[1]).collect(), dtype="float32")
        X_validation = np.array(validation_data_rdd.map(lambda x: x[0]).collect(), dtype="float32")

        model = XGBRegressor()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)

        prediction_data = [[row[0], row[1], pred] for row, pred in zip(validation_data_rdd.collect(), predictions)]
        save_predictions(prediction_data, output_file)

        execution_duration = time.time() - start_time
        print(f"Execution Time: {execution_duration:.2f} seconds")

    finally:
        sc.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit script.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)

    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    run_task(folder_path, test_file_name, output_file_name)
