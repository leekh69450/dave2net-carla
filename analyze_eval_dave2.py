import argparse
import pandas as pd

def main(args):
    df = pd.read_csv(args.csv)

    # Filter failed spawns
    df = df[df["spawn_failed"] == 0]

    # Success condition
    df["success"] = (
        (df["distance_m"] >= args.success_dist) &
        (df["collision"] == 0) &
        (df["stuck"] == 0)
    )

    print("CSV:", args.csv)
    print("Episodes:", len(df))
    print("Success rate:", df["success"].mean())

    print("\nDistance stats:")
    print(df["distance_m"].describe())

    print("\nFailure rates:")
    print("Collision rate:", df["collision"].mean())
    print("Stuck rate:", df["stuck"].mean())
    print("Avg lane invasions:", df["lane_invasions"].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze CARLA evaluation results"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to evaluation CSV file"
    )

    parser.add_argument(
        "--success-dist",
        type=float,
        default=200.0,
        help="Distance threshold for success (meters)"
    )

    args = parser.parse_args()
    main(args)
