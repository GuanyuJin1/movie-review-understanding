import argparse

from src.movie_review_understanding.demo.cli_demo import run_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the movie review understanding demo.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the optional LLM classification step for faster CPU-only runs.",
    )
    parser.add_argument(
        "--llm-sample-size",
        type=int,
        default=None,
        help="Number of test reviews to sample for each LLM prompt style.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(skip_llm=args.skip_llm, llm_sample_size=args.llm_sample_size)
