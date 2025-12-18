from preprocessing import ESNLIPreprocessor
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

def _extract_path(f: str) -> Tuple[Path, str]:
    split_filename = f.split('/')
    path, name = split_filename[:-1], split_filename[-1]
    path = Path("/".join(path))
    return path, name


def main():

    argparser = ArgumentParser(
        prog="Data Preprocessing Pipeline for Structured e-SNLI",
        description="Run the preprocessing pipeline to obtained clean csv files ready for generating structured explanations",
        epilog="LoLa Project"
    )
    argparser.add_argument('filename')

    args = argparser.parse_args()

    path, name = _extract_path(args.filename)

    df = pd.read_csv(args.filename)

    preprocessor = ESNLIPreprocessor(df)
    preprocessed_data = preprocessor.extract_highlighted_words()
    preprocessed_data = preprocessor.add_sentence_lengths()
    preprocessed_data = preprocessor.count_highlighted_words()
    preprocessed_data = preprocessor.create_ordered_highlights_as_list()
    cleaned_data = preprocessor.cleanup_and_restructure()

    preprocessor.data.to_csv(path / ("preprocessed_" + name), index=False)
    preprocessor.cleaned_data.to_csv(path / ("cleaned_" + name), index=False)

if __name__ == '__main__':
    main()