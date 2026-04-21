#!/usr/bin/env python3
"""
Calculate extraction success rates for new fields in Douyin video metadata.
"""
import csv
import sys
from pathlib import Path

# New fields to analyze
NEW_FIELDS = [
    'author_follower_count',
    'author_total_favorited',
    'author_signature',
    'author_verification_type',
    'video_cover_url',
    'dynamic_cover_url',
    'origin_cover_url'
]

def analyze_csv(file_path: Path):
    """Analyze CSV file and calculate extraction success rates."""
    total_records = 0
    high_confidence_records = 0
    field_counts = {field: 0 for field in NEW_FIELDS}

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_records += 1

            # Check if record is high confidence match
            match_type = row.get('match_type', '')
            confidence = row.get('confidence', '')

            if match_type == 'exact' and confidence == 'high':
                high_confidence_records += 1

                # Count non-empty values for each new field
                for field in NEW_FIELDS:
                    value = row.get(field, '')
                    # Check if value is not empty (not None, not empty string)
                    if value is not None and str(value).strip() != '':
                        field_counts[field] += 1

    # Calculate success rates
    success_rates = {}
    for field in NEW_FIELDS:
        if high_confidence_records > 0:
            rate = field_counts[field] / high_confidence_records * 100
        else:
            rate = 0.0
        success_rates[field] = rate

    return {
        'total_records': total_records,
        'high_confidence_records': high_confidence_records,
        'field_counts': field_counts,
        'success_rates': success_rates
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python field_extraction_stats.py <csv_file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    results = analyze_csv(file_path)

    print("=" * 60)
    print("FIELD EXTRACTION SUCCESS RATE ANALYSIS")
    print("=" * 60)
    print(f"Total records in CSV: {results['total_records']}")
    print(f"High confidence matches (exact/high): {results['high_confidence_records']}")
    print(f"Success rate base: {results['high_confidence_records']} records")
    print("-" * 60)

    # Print field statistics
    print("Extraction Success Rates:")
    print("-" * 60)
    for field in NEW_FIELDS:
        count = results['field_counts'][field]
        rate = results['success_rates'][field]
        print(f"{field:30s}: {count:3d}/{results['high_confidence_records']:3d} = {rate:6.2f}%")

    print("-" * 60)

    # Summary statistics
    avg_success_rate = sum(results['success_rates'].values()) / len(NEW_FIELDS)
    print(f"Average success rate across all new fields: {avg_success_rate:.2f}%")

    # Identify fields with less than 100% success
    low_fields = [(field, rate) for field, rate in results['success_rates'].items() if rate < 100]
    if low_fields:
        print("\nFields with less than 100% extraction rate:")
        for field, rate in low_fields:
            print(f"  - {field}: {rate:.2f}%")
    else:
        print("\nAll fields have 100% extraction rate!")

    print("=" * 60)

if __name__ == '__main__':
    main()