
import csv
import re
from pathlib import Path

# Dataset folder
folder = Path(
    r"C:\Users\veere\Downloads\dataset\Scenario B-ARFF"
)

print("=" * 55)
print("ARFF TO CSV CONVERSION")
print("=" * 55)

# Check folder
if not folder.exists():
    print("ERROR: Dataset folder not found!")
    input("Press Enter to exit...")
    raise SystemExit

# Find all standard Scenario B ARFF files
arff_files = sorted([
    file for file in folder.glob("TimeBasedFeatures-Dataset-*.arff")
    if "AllinOne" not in file.name
])

if not arff_files:
    print("ERROR: No ARFF files found!")
    input("Press Enter to exit...")
    raise SystemExit

# Process each file
for arff_file in arff_files:

    print(f"\nProcessing: {arff_file.name}")

    try:
        lines = arff_file.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

        # Extract column names
        columns = []

        for line in lines:
            match = re.match(
                r"""^\s*@ATTRIBUTE\s+(?:'([^']+)'|"([^"]+)"|(\S+))""",
                line,
                re.IGNORECASE
            )

            if match:
                column = next(
                    value for value in match.groups()
                    if value is not None
                )
                columns.append(column)

        if not columns:
            print("ERROR: No column names found.")
            continue

        # Find the DATA section
        data_index = None

        for i, line in enumerate(lines):
            if line.strip().upper().startswith("@DATA"):
                data_index = i
                break

        if data_index is None:
            print("ERROR: @DATA section not found.")
            continue

        # Create CSV file
        csv_file = arff_file.with_suffix(".csv")

        total_records = 0
        skipped_rows = 0
        class_counts = {}

        with csv_file.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as output:

            writer = csv.writer(output)

            # Write column names
            writer.writerow(columns)

            # Read dataset records
            for line_number, line in enumerate(
                lines[data_index + 1:],
                start=data_index + 2
            ):

                line = line.strip()

                # Skip blank lines and comments
                if not line or line.startswith("%"):
                    continue

                values = next(csv.reader([line]))

                # Skip rows containing only commas or empty values
                if not any(value.strip() for value in values):
                    skipped_rows += 1
                    continue

                # Remove empty values at the end
                while values and values[-1].strip() == "":
                    values.pop()

                values = [value.strip() for value in values]

                # Check number of columns
                if len(values) != len(columns):
                    raise ValueError(
                        f"Problem at line {line_number}: "
                        f"Expected {len(columns)} columns, "
                        f"but found {len(values)}."
                    )

                # Write record
                writer.writerow(values)
                total_records += 1

                # Count traffic classes
                label = values[-1]
                class_counts[label] = (
                    class_counts.get(label, 0) + 1
                )

        # Display conversion results
        print(f"SUCCESS: {csv_file.name}")
        print(f"Records: {total_records}")
        print(f"Columns: {len(columns)}")
        print(f"Empty rows skipped: {skipped_rows}")

        print("\nTraffic class distribution:")

        for label, count in sorted(class_counts.items()):
            print(f"{label:20} : {count}")

    except Exception as error:
        print(f"ERROR converting {arff_file.name}")
        print(error)

print("\n" + "=" * 55)
print("CONVERSION PROCESS COMPLETED")
print("=" * 55)
