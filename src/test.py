import random
import csv

def generate_data(center=1.7, tolerance=0.05, total=100, outlier_fraction=0.1):
    data = []

    # Number of outliers
    outliers_count = int(total * outlier_fraction)

    # Generate inliers
    for _ in range(total - outliers_count):
        value = random.uniform(center - tolerance, center + tolerance)
        data.append(round(value, 4))

    # Generate outliers
    for _ in range(outliers_count):
        offset = random.uniform(0.01, 0.07)
        direction = random.choice([-1, 1])
        value = center + direction * offset
        data.append(round(value, 4))

    random.shuffle(data)
    return data

# Generate the data
values = generate_data()

# Write to CSV
with open("tolerance_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["value"])
    for v in values:
        writer.writerow([v])

print("CSV file 'tolerance_data.csv' saved successfully.")
