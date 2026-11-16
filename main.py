import time
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assets")

script_start = time.perf_counter()


spark = (
    SparkSession.builder.master("local[*]")
    .appName("first-look")
    .config("spark.driver.memory", "8g")
    .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

def speed_alerts(df, limit):
    return (df.filter(F.col("value") > limit)
        .select("vehicle_id", "timestamp",
                F.lit("SPEED_LIMIT_EXCEEDED").alias("anomaly_type"),
                F.concat(F.lit("Speed "), F.round("value", 1).cast("string"),
                         F.lit(f" m/s exceeds the configured limit of {limit} m/s")).alias("description")))

def gap_alerts(df, minutes):
    w = Window.partitionBy("vehicle_id").orderBy("timestamp")
    return (df.select("vehicle_id", "timestamp").distinct()
        .withColumn("prev_ts", F.lag("timestamp").over(w))
        .withColumn("gap_sec",
                    F.expr("timestampdiff(MICROSECOND, prev_ts, timestamp)") / 1000000)
        .filter(F.col("gap_sec") > minutes * 60)
        .select("vehicle_id", "timestamp",
                F.lit("COMMUNICATION_GAP").alias("anomaly_type"),
                F.concat(F.lit("No data for "), F.round(F.col("gap_sec") / 60, 1).cast("string"),
                         F.lit(f" min (threshold {minutes} min), since "),
                         F.col("prev_ts").cast("string")).alias("description")))

signals = spark.read.parquet(f"{BASE}/dim_signal_info")

factDf = spark.read.option("mergeSchema", "true").parquet(f"{BASE}/fact_signal_data")


print("\nFact table columns and types:")
factDf.printSchema()

factDf.show(truncate=False)

#Number of vehicles
veh_number = factDf.select("vehicle_id").distinct().count()
print("Number of vehicles: ", veh_number)

#Total number of records
total_records = factDf.count()
print("Total records: ", total_records)

#Time span
factDf.agg(F.min("timestamp").alias("Start date"), F.max("timestamp").alias("End date")).show(truncate=False)

#Data cleaning
#Create a single value column with either the float or int value, discard string
factDf = (
    factDf
    .drop("_ducklake_internal_row_id", "_ducklake_internal_snapshot_id", "string_value")
    .withColumn("value", F.coalesce(F.col("float_value"), F.col("int_value").cast("double")))
    .drop("float_value", "int_value")
)

#Condition to track speed readings
is_speed = F.col("signal_id") == 680

cleanDf = (
    factDf
    .na.drop(subset=["timestamp", "signal_id", "session_id", "value"])
    .filter(~(is_speed & (F.col("value") < 0)))
    .dropDuplicates(["session_id", "signal_id", "timestamp"])
).cache()
cleanDf.count()

# Records per day and per hour
print("Count per day:")
cleanDf.groupBy(F.to_date("timestamp").alias("day")).count().show()
print("Count per hour:")
cleanDf.groupBy(F.date_trunc("hour", "timestamp").alias("hour")).count().orderBy("hour").show()



print("Missing value:", cleanDf.filter(F.col("value").isNull()).count())
print("Missing timestamp:", cleanDf.filter(F.col("timestamp").isNull()).count())
print("Negative speed:", cleanDf.filter(is_speed & (F.col("value") < 0)).count())

#Duplicate check based on combination of timestamp, session and signal
key = ['timestamp', 'session_id', 'signal_id']
duplicates = total_records - cleanDf.dropDuplicates(key).count()
print("Duplicate rows:", duplicates)

#Data analysis
#Records per vehicle
print("Records per vehicle: ")
cleanDf.groupBy('vehicle_id', 'vehicle_name').count().show()

print("Vehicle speed distribution: ")
(cleanDf.filter(is_speed)
    .groupBy("vehicle_name")
    .agg(F.min("value").alias("min"),
         F.expr("percentile_approx(value, 0.5)").alias("median"),
         F.expr("percentile_approx(value, 0.99)").alias("p99"),
         F.max("value").alias("max"))
    .show(truncate=False))

#Speed per vehicle
print("Speed per vehicle: ")
cleanDf.filter(is_speed).groupBy('vehicle_name').agg(F.avg("value").alias("average_speed"),F.max("value").alias("maximum_speed") ).show()

#Speed limit exceed
print("Speed limit exceed per vehicle: ")
speed_limit = 25                       #90 kmh
cleanDf.filter(is_speed).filter(F.col("value") > speed_limit).groupBy("vehicle_name").count().show()

max_plausible_speed = 70   #m/s (250 kmh)

median_speed = (cleanDf.filter(is_speed).groupBy("vehicle_id", "vehicle_name")
                .agg(F.expr("percentile_approx(value, 0.5)").alias("median_speed")))
bad = median_speed.filter(F.col("median_speed") > max_plausible_speed)
bad_ids = [r["vehicle_id"] for r in bad.collect()]
print("Excluded from speed analysis:", bad_ids)

#Gas pedal usage
gas_signal = 157
full_throttle = 0.99          # pedal position treated as 100% (scale is 0-1)
full_throttle_seconds = 5     # minimum duration for a full throttle alert

is_gas = F.col("signal_id") == gas_signal
gasDf = cleanDf.filter(is_gas)

#Gas per vehicle
print("Gas pedal per vehicle: ")
gasDf.groupBy("vehicle_name").agg(
    F.avg("value").alias("average_gas"), F.max("value").alias("maximum_gas")).show()

#Full throttle readings per vehicle
print("Full throttle readings per vehicle: ")
gasDf.filter(F.col("value") >= full_throttle).groupBy("vehicle_name").count().show()

#Speed dataframe
speedDf = cleanDf.filter(is_speed & ~F.col("vehicle_id").isin(bad_ids))

avg_speed_pd = (
    speedDf.groupBy("vehicle_name")
    .agg(F.avg("value").alias("average_speed"))
    .orderBy("vehicle_name")
    .toPandas()
)

#Violations per vehicle
violations_pd = (
    speedDf.groupBy("vehicle_name")
    .agg(F.sum(F.when(F.col("value") > speed_limit, 1).otherwise(0)).alias("violations"))
    .orderBy("vehicle_name")
    .toPandas()
)

avg_gas_pd = (gasDf.groupBy("vehicle_name")
              .agg(F.avg("value").alias("average_gas"))
              .orderBy("vehicle_name").toPandas())

full_gas_pd = (gasDf.groupBy("vehicle_name")
               .agg(F.sum(F.when(F.col("value") >= full_throttle, 1).otherwise(0)).alias("full_throttle_readings"))
               .orderBy("vehicle_name").toPandas())

def bar_plot(df, ycol, ylabel, title, color, filename, fmt="%.1f"):
    n = len(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(df["vehicle_name"], df[ycol], width=0.35, color=color)
    ax.set_xlim(-0.8, n - 1 + 0.8)
    ax.set_xlabel("Vehicle")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.bar_label(bars, fmt=fmt, padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

bar_plot(avg_speed_pd, "average_speed", "Average speed (m/s)",
         "Average speed per vehicle", "steelblue", "avg_speed_per_vehicle.png")

bar_plot(violations_pd, "violations", f"Readings above {speed_limit} m/s",
         "Speed limit violations per vehicle", "indianred", "speed_violations_per_vehicle.png")

#Gas plots
bar_plot(avg_gas_pd, "average_gas", "Average gas pedal position (0-1)",
         "Average gas pedal position per vehicle", "seagreen", "avg_gas_per_vehicle.png", fmt="%.2f")

bar_plot(full_gas_pd, "full_throttle_readings", f"Readings at or above {full_throttle}",
         "Full throttle readings per vehicle", "darkorange", "full_throttle_per_vehicle.png", fmt="%.0f")

#Communication gap threshold
gap_minutes = 10

#Idling threshold
idle_minutes = 10

# m/s, below this the vehicle counts as stationary
idle_speed = 0.5


#Speed limit exceed report
speed_alert_df = speed_alerts(cleanDf.filter(is_speed), speed_limit)

#Full throttle: group consecutive readings at full throttle into runs
wg = Window.partitionBy("vehicle_id", "session_id").orderBy("timestamp")

gas_runs = (
    gasDf
    .withColumn("full", F.col("value") >= full_throttle)
    .withColumn("changed", (F.col("full") != F.lag("full").over(wg)).cast("int"))
    .withColumn("run_id", F.sum(F.coalesce("changed", F.lit(0))).over(wg))
)

gas_alerts = (
    gas_runs.filter("full")
    .groupBy("vehicle_id", "session_id", "run_id")
    .agg(F.min("timestamp").alias("start_ts"), F.max("timestamp").alias("end_ts"))
    .withColumn("dur_sec", F.expr("timestampdiff(MILLISECOND, start_ts, end_ts)") / 1000)
    .filter(F.col("dur_sec") >= full_throttle_seconds)
    .select(
        "vehicle_id",
        F.col("start_ts").alias("timestamp"),
        F.lit("FULL_THROTTLE").alias("anomaly_type"),
        F.concat(
            F.lit("Gas pedal at full throttle for "), F.round("dur_sec", 1).cast("string"),
            F.lit(f" s (threshold {full_throttle_seconds} s)"),
        ).alias("description"),
    )
)


#Communication gap
w = Window.partitionBy("vehicle_id").orderBy("timestamp")

gap_alert_df = gap_alerts(cleanDf, gap_minutes)


#Idle sessions
ws = Window.partitionBy("vehicle_id").orderBy("timestamp")

runs = (
    cleanDf.filter(is_speed)
    .withColumn("stationary", F.col("value") < idle_speed)
    .withColumn("changed", (F.col("stationary") != F.lag("stationary").over(ws)).cast("int"))
    .withColumn("run_id", F.sum(F.coalesce("changed", F.lit(0))).over(ws))
)

idle_alerts = (
    runs.filter("stationary")
    .groupBy("vehicle_id", "run_id")
    .agg(F.min("timestamp").alias("start_ts"), F.max("timestamp").alias("end_ts"))
    .withColumn("dur_min", F.expr("timestampdiff(SECOND, start_ts, end_ts)") / 60)
    .filter(F.col("dur_min") > idle_minutes)
    .select(
        "vehicle_id",
        F.col("start_ts").alias("timestamp"),
        F.lit("PROLONGED_IDLING").alias("anomaly_type"),
        F.concat(
            F.lit("Stationary for "), F.round("dur_min", 1).cast("string"),
            F.lit(f" min (threshold {idle_minutes} min)"),
        ).alias("description"),
    )
)

#Combine alerts
alerts = speed_alert_df.unionByName(gap_alert_df).unionByName(gas_alerts).unionByName(idle_alerts)

#Store alerts as one readable CSV with the names used in the assignment
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT, exist_ok=True)

alerts_pd = (alerts
    .withColumnRenamed("vehicle_id", "VehicleID")
    .withColumnRenamed("timestamp", "Timestamp")
    .withColumnRenamed("anomaly_type", "AnomalyType")
    .withColumnRenamed("description", "Description")
    .orderBy("VehicleID", "Timestamp")
    .toPandas())
alerts_pd.to_csv(os.path.join(OUTPUT, "alerts.csv"), index=False, encoding="utf-8-sig")
print(f"Saved {len(alerts_pd)} alerts to {OUTPUT}")

#Show anomalies from pandas dataframe, to avoid recalculating the Spark dataframe with show()
print(alerts_pd["AnomalyType"].value_counts().to_string())

#Measure time between two Spark queries, one for vehicle speed and one for lag
def timed(label, df, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        df.collect()
        times.append(time.perf_counter() - t0)
    print(f"{label}: " + ", ".join(f"{t:.2f}s" for t in times))

timed("Query 1: Average speed per vehicle", speedDf.groupBy("vehicle_name").agg(F.avg("value")))
timed("Query 2: Communication gaps (lag window)", gap_alert_df)

elapsed = time.perf_counter() - script_start
print(f"\nTotal execution time: {elapsed:.2f} s ({elapsed/60:.2f} min)")


#Streaming
HERE = os.path.dirname(os.path.abspath(__file__))
WATCHED = os.path.join(HERE, "stream_input")
CHECKPOINT = os.path.join(HERE, "checkpoint")
OUT_CSV = os.path.join(OUTPUT, "stream_alerts.csv")

feed = (speedDf.select("vehicle_id", "timestamp", "value")
        .orderBy("timestamp").limit(30000).toPandas())
feed["vehicle_id"] = feed["vehicle_id"].astype(str)

#Synthetic speed readings
n_files = 10
chunk_size = len(feed) // n_files
fake_rows = [int(len(feed) * p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
feed.loc[fake_rows, "vehicle_id"] = "SYNTH-STREAM"
feed.loc[fake_rows, "value"] = [30.0, 32.5, 35.0, 29.0, 40.0]
print("NOTE: readings with vehicle_id SYNTH-STREAM are synthetic test data")

for p in (WATCHED, CHECKPOINT):
    shutil.rmtree(p, ignore_errors=True)
os.makedirs(WATCHED)
if os.path.exists(OUT_CSV): os.remove(OUT_CSV)

stream = (spark.readStream
          .schema("vehicle_id STRING, timestamp TIMESTAMP, value DOUBLE")
          .option("header", "true").option("maxFilesPerTrigger", 1)
          .csv(WATCHED))

def handle(batch_df, batch_id):
    pdf = batch_df.toPandas()
    print(f"batch {batch_id}: {len(pdf)} alerts")
    if len(pdf):
        pdf.to_csv(OUT_CSV, mode="a", index=False, header=not os.path.exists(OUT_CSV))

query = (speed_alerts(stream, speed_limit).writeStream
         .foreachBatch(handle).option("checkpointLocation", CHECKPOINT).start())

for i in range(n_files):
    tmp = os.path.join(WATCHED, "_writing.csv")
    feed.iloc[i * chunk_size:(i + 1) * chunk_size].to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(WATCHED, f"chunk_{i}.csv"))
    print(f"File {i + 1}/{n_files} arrived")
    time.sleep(2)

query.processAllAvailable()
query.stop()


saved = pd.read_csv(OUT_CSV)

#Evaluation with manually created test set
rows = [
    ("SYNTH-A", "2099-01-01 08:00:00", 10.0),
    ("SYNTH-A", "2099-01-01 08:00:10", 20.0),
    ("SYNTH-A", "2099-01-01 08:00:20", 26.0),
    ("SYNTH-A", "2099-01-01 08:00:30", 25.0),
    ("SYNTH-A", "2099-01-01 08:00:40", 31.5),   #alert
    ("SYNTH-A", "2099-01-01 08:21:40", 12.0),   #alert
    ("SYNTH-A", "2099-01-01 08:21:50", 24.9),
    ("SYNTH-A", "2099-01-01 08:31:50", 15.0),
    ("SYNTH-A", "2099-01-01 08:47:50", 18.0),   #alert
    ("SYNTH-B", "2099-01-01 08:00:00", 5.0),
    ("SYNTH-B", "2099-01-01 08:05:00", 40.0),
    ("SYNTH-B", "2099-01-01 08:15:30", 8.0),    #alert
]
test = (spark.createDataFrame(rows, ["vehicle_id", "ts", "value"])
        .withColumn("timestamp", F.to_timestamp("ts")).drop("ts"))

def keys(df):
    return {(r["vehicle_id"], r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")) for r in df.collect()}

failed = False
def check(name, got, expected):
    global failed
    ok = got == expected
    failed |= not ok
    print(f"{name}: " + ("PASS" if ok else f"FAIL (missing {expected - got}, unexpected {got - expected})"))

sp = speed_alerts(test, speed_limit_test := 25)
gp = gap_alerts(test, 10)

check("Speed check", keys(sp), {("SYNTH-A", "2099-01-01 08:00:20"),
                                ("SYNTH-A", "2099-01-01 08:00:40"),
                                ("SYNTH-B", "2099-01-01 08:05:00")})
check("Gap check", keys(gp), {("SYNTH-A", "2099-01-01 08:21:40"),
                              ("SYNTH-A", "2099-01-01 08:47:50"),
                              ("SYNTH-B", "2099-01-01 08:15:30")})

if failed:
    raise SystemExit(1)