import logging
import os
from distutils.file_util import move_file, copy_file

import pandas as pd
from dotenv import load_dotenv

from classifier.llm_classifier import LLMDataExtractor
from classifier.pdfprocessor import PdfProcessor
from classifier.renamer import *

load_dotenv()

logging.debug(os.environ.items())

PROJECT_ROOT=Path(os.getenv("PROJECT_ROOT", "."))


def parse_cmdline():
    """parses the commandline and identifies:

    --pdf-in [path] : folder of pdfs to classify
    --pdf-out [path] : folder to move classified pdfs to
    --move : if set, the pdf will be moved, if not, nothing happens
    --results : if set, a results.csv will be written to the pdf-out
    --results-name [name] : the name of the results.csv

    :returns
        the parsed arguments
    """

    import argparse
    parser = argparse.ArgumentParser(description='Classify PDF files.')
    parser.add_argument('--pdf-in', type=Path, help='Folder of PDFs to classify', required=True)
    parser.add_argument('--pdf-out', type=Path, help='Folder to move classified PDFs to', required=True)
    parser.add_argument('--move', action='store_true', default=False, help='Move PDFs to the output folder')
    parser.add_argument('--copy', action='store_true', default=False, help='Copy PDFs to the output folder')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Do a dry run, neither copy nor move are executed, results and features are written, though')
    parser.add_argument('--force', action='store_true', default=False, help='Force overwrite of existing data files')
    parser.add_argument('--results', action='store_true', default=True, help='Write a results.csv file')
    parser.add_argument('--results-name', type=str, default='results.csv', help='Name of the results.csv file')
    parser.add_argument('--features', action='store_true', default=True, help='Write a features.csv file')
    parser.add_argument('--features-name-fmt', type=str, default='{pdf_name}-feature.csv', help='format string for the feature file, use {pdf_name} for the name of the accompanying pdf')
    args = parser.parse_args()
    return args


def main():
    """
    Main function to classify PDF files.

    """

    args = parse_cmdline()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    pdf_path = args.pdf_in
    results_file = pdf_path / args.results_name
    target_base = args.pdf_out
    target_base.mkdir(parents=True, exist_ok=True)
    target_workdir = target_base / "work.d"
    target_workdir.mkdir(parents=True, exist_ok=True)
    write_features = args.features
    features_name_fmt = args.features_name_fmt
    write_results = args.results
    execute_move = args.move
    execute_copy = args.copy
    do_dry_run = args.dry_run
    do_force = args.force

    if execute_move and execute_copy :
        logging.error("can either move or copy, not both, use --copy or --move, not both")
        exit(-1)
# tag::main-method-init-llm[]
    llm_c = LLMDataExtractor(api_key=os.getenv("GOOGLE_API_KEY"), model_name="gemini-2.0-flash-lite")
# end::main-method-init-llm[]
# tag::main-method-process-pdf[]
    pdf_data_list = PdfProcessor(target_workdir, do_force).process_pdfs(pdf_path)
# end::main-method-process-pdf[]
    features_list = []

    _features_changed = False
    for pdf_data in pdf_data_list:
        logging.info(f"Found: {pdf_data.pdf_file.name}")
        _feature_name = features_name_fmt.format(pdf_name=pdf_data.pdf_file.name)
        _feature_path = target_base / _feature_name

        # use the existing feature file if it exists and is newer than the pdf file and not forced
        if not do_force and _feature_path.exists() and os.path.getmtime(_feature_path) > os.path.getmtime(pdf_data.pdf_file):
            logging.info(f"Using existing features from {_feature_path}")
            _existing:pd.DataFrame = pd.read_csv(_feature_path, index_col=0)
            features_list.append(_existing)
            continue

        _features_changed = True

# tag::main-method-loop-features[]
        best_avg = 0.0
        best = None
        for data in pdf_data.data:
            features = llm_c.extract_features(data)
            features_avg = features["quality"].mean()
            if features_avg > best_avg :
                best = features
                best_avg = features_avg
# end::main-method-loop-features[]

        if best is not None:
            pdf_id_row = pd.DataFrame({'key':['id'], 'value':[pdf_data.pdf_file.absolute()], 'quality':[1.0]})
            pdf_id_row.set_index("key", inplace=True)
            best.set_index("key", inplace=True)
            best = pd.concat([pdf_id_row, best], ignore_index=False)
            if write_features :
                logging.info(f"Writing features to {_feature_path}")
                best.to_csv(_feature_path, index=True)
            features_list.append(best)

    # writing features_list to features.csv next to input pdfs
    if write_features:
        features_df = pd.concat(features_list)
        # Keep existing features compilation if they exist and are unchanged and not forced
        if not do_force and (target_base / "all-features.csv").exists() and not _features_changed:
            logging.info(f"Keeping existing features from {target_base / 'all-features.csv'}")
        else:
            logging.info(f"Writing features compilation to {target_base} / all-features.csv")
            features_df.to_csv(target_base / "all-features.csv", index=True, header=True)

# tag::main-method-sanitize-data[]
    classification_data = {feat.loc["id", "value"]: FileData().init_from_features(feat) for feat in features_list}
    sanitized_data = sanitize_filename_data(classification_data)
# end::main-method-sanitize-data[]
    if write_results :
        logging.info(f"Writing results to {results_file}")
        sanitized_data_list = [vars(v) for v in sanitized_data.values()]
        pd.DataFrame(sanitized_data_list,
                     columns = ['docdate', 'doctype', 'sendername', 'docid', 'receivername', 'dateoffile', 'extension', 'id']
                     ).to_csv(results_file, index=False)

# tag::main-method-classify-pdf[]
    for name, _file_data in sanitized_data.items():
        new_path = classify_pdf(_file_data, target_base)
        if new_path:
            print(f"Classified: {_file_data.id} -> {new_path.name}")
            if execute_move: move_file(_file_data.id, new_path, verbose=1, dry_run=do_dry_run )
            if execute_copy: copy_file(_file_data.id, new_path, verbose=1, dry_run=do_dry_run )
        else:
            print(f"No classification data found for: {_file_data.id}")
# end::main-method-classify-pdf[]

if __name__ == "__main__":
    main()
