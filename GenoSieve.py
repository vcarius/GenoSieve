import os
import math
from math import comb
import random

import json

import pandas as pd 
import numpy as np
from Bio import SeqIO
from Bio import Phylo

from datetime import datetime

import warnings
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict, Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances

# Use scipy to create a Compressed Sparse Row (CSR) matrix, which is highly efficient
from scipy.sparse import csr_matrix

from numpy.linalg import slogdet

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps

from tqdm import tqdm
import argparse
import logging


def save_dict2json(Dict: object = None, filename: str = None):
        
    def _make_jsonable(obj):
        # pandas Series -> dict
        if pd is not None and isinstance(obj, pd.Series):
            obj = obj.to_dict()

        # pandas DataFrame -> dict(index_str -> {col_str: val})
        if pd is not None and isinstance(obj, pd.DataFrame):
            return {str(idx): {str(col): _make_jsonable(obj.at[idx, col]) for col in obj.columns} for idx in obj.index}

        # dict: convert keys to str (safe) and values recursively
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # force key to string (Period, Timestamp, custom objects, etc.)
                # JSON only accepts str,int,float,bool,None as keys; to be safe, use str(key)
                key_str = k if isinstance(k, (str, int, float, bool, type(None))) else str(k)
                out[str(key_str)] = _make_jsonable(v)
            return out

        # sequences -> list
        if isinstance(obj, (list, tuple, set)):
            return [_make_jsonable(x) for x in obj]

        # numpy scalars / arrays
        if np is not None:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                f = float(obj)
                return None if (math.isnan(f) or math.isinf(f)) else f
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return [_make_jsonable(x) for x in obj.tolist()]

        # pandas / numpy NA
        try:
            if pd is not None and pd.isna(obj):
                return None
        except Exception:
            pass

        # pandas Period / Timestamp / Timedelta -> str
        if pd is not None and isinstance(obj, (pd.Period, pd.Timestamp, pd.Timedelta)):
            return str(obj)

        # datetime -> ISO string
        if isinstance(obj, datetime):
            return obj.isoformat()

        # primitive Python types (with NaN/Inf guard for float)
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, (int, str, bool)) or obj is None:
            return obj

        # fallback: stringify
        return str(obj)

    serializable = _make_jsonable(Dict)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)

def error_calculate(df1: pd.DataFrame = None, df2: pd.DataFrame = None):
    
    all_index = df1.index.union(df2.index)
    all_cols = df1.columns.union(df2.columns)
    df1 = df1.reindex(index=all_index, columns=all_cols, fill_value=0.0)
    df2 = df2.reindex(index=all_index, columns=all_cols, fill_value=0.0)
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    diff = (df1 - df2)
    abs_diff = abs(diff)
    squared_diff = diff ** 2

    mae_per_clade = abs_diff.mean(axis=1).to_dict()
    mae_per_date = abs_diff.mean(axis=0).to_dict()
    mae_overall = abs_diff.values.mean() if abs_diff.size > 0 else np.nan
    
    mae = {'mae_per_clade': mae_per_clade,
            'mae_per_date': mae_per_date,
            'mae_overall': mae_overall,
    }

    mse_per_clade = squared_diff.mean(axis=1).to_dict()
    mse_per_date = squared_diff.mean(axis=0).to_dict()
    mse_overall = squared_diff.values.mean() if squared_diff.size > 0 else np.nan
    
    mse = {'mse_per_clade': mse_per_clade,
            'mse_per_date': mse_per_date,
            'mse_overall': mse_overall,
    }

    return mae, mse

def streamPlot(df: pd.DataFrame=None, clade_color_dict: dict = None, output: str = None, figsize=(24, 8), date_freq='M'):

    plt.figure(figsize=figsize)
    
    # Initialize a variable to keep track of the maximum frequency value
    max_freq_value = 0

    # Set the title of the plot
    plt.title('Clade Frequencies', fontsize=20)

    df2 = df[['date', 'clade']].copy()
    df2['date'] = pd.to_datetime(df2['date'], errors='coerce')
    df2['date_freq'] = df2['date'].dt.to_period(date_freq)

    # Group the data by month and lineage, calculate the frequency, and reshape the DataFrame
    freq_by_date = df2.groupby(['date_freq', 'clade']).size().unstack(fill_value=0)
    freq_by_date = freq_by_date.T/freq_by_date.sum(axis=1)
    freq_by_date_percentage = freq_by_date.T * 100

    # Update the maximum frequency value
    max_freq_value = max(max_freq_value, freq_by_date_percentage.values.max())

    # Extract the days for the x-axis
    days = [str(month) for month in freq_by_date_percentage.index]

    # Assign colors to each clade for plotting
    clade_colors=[clade_color_dict[c] for c in freq_by_date_percentage.columns]


    # Plot the data as a stacked area plot
    lines = plt.stackplot(days, freq_by_date_percentage.T.values, labels=freq_by_date_percentage.columns, colors =clade_colors)
    plt.xlabel('Date')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Frequency (%)')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=len(set(df.clade)), fontsize='14')
    plt.ylim(0, 100)

    # Save the plot as a PDF file
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()

    return freq_by_date

def READ_METADATA(TSV_FILE: str = None) -> pd.DataFrame:
    """
    Reads a TSV metadata file and ensures that the required columns
    'name', 'date', 'region', and 'clade' are present.
    
    Parameters
    ----------
    TSV_FILE : str
        Path to the TSV metadata file.
    
    Returns
    -------
    pd.DataFrame
        Loaded and validated DataFrame.
    
    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if TSV_FILE is None:
        raise ValueError("A TSV file path must be provided.")

    required_columns = {'name', 'date', 'region', 'clade'}
    df = pd.read_csv(TSV_FILE, sep="\t")

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {', '.join(missing)}")

    print("✔ All required metadata columns are present.")
    return df

def REMOVE_REDUNDANCY(fasta_dict: Dict[str, str],
                      date_format: str = "%Y-%m-%d",
                      keep: str = "oldest") -> Dict[str, str]:
    """
    Remove redundant sequences from a dict mapping FASTA description -> sequence.
    If multiple IDs have the same sequence, keeps either the 'oldest' or the 'newest'
    according to a date parsed from the last '|'-separated field of the ID.

    Parameters
    ----------
    fasta_dict : Dict[str, str]
        Mapping from FASTA description (ID string) to sequence string.
    date_format : str, optional
        The expected datetime format in the ID (default: "%Y-%m-%d").
        If parsing fails, string comparison is used as a fallback.
    keep : {'oldest','newest'}, optional
        Whether to keep the 'oldest' (default) or 'newest' sequence when duplicates are found.

    Returns
    -------
    Dict[str, str]
        Non-redundant mapping ID -> sequence (same format as input).
    """
    if fasta_dict is None:
        raise ValueError("fasta_dict must be provided (dict of ID -> sequence).")

    if keep not in {"oldest", "newest"}:
        raise ValueError("keep must be 'oldest' or 'newest'.")

    # Map seq -> (chosen_id, parsed_date_or_none, raw_date_string)
    seq_index = {}

    for id_str, seq in fasta_dict.items():
        if seq is None:
            continue

        raw_date = id_str.split("|")[-1].strip()
        parsed_date = None
        try:
            parsed_date = datetime.strptime(raw_date, date_format)
        except Exception:
            # fallback: parsed_date stays None and we will use raw_date string for comparison
            pass

        if seq in seq_index:
            prev_id, prev_date, prev_raw = seq_index[seq]
            # comparator: if both parsed, compare datetimes; else compare raw_date strings
            def is_current_preferred(current_date, prev_date, current_raw, prev_raw):
                if current_date is not None and prev_date is not None:
                    if keep == "oldest":
                        return current_date < prev_date
                    else:
                        return current_date > prev_date
                # fallback to string comparison
                if keep == "oldest":
                    return current_raw < prev_raw
                return current_raw > prev_raw

            if is_current_preferred(parsed_date, prev_date, raw_date, prev_raw):
                seq_index[seq] = (id_str, parsed_date, raw_date)
            else:
                # keep previous; optionally print notification
                print(f"Duplicate sequence found: keeping {seq_index[seq][0]} over {id_str}.")
        else:
            seq_index[seq] = (id_str, parsed_date, raw_date)

    # Reconstruct dict ID -> sequence (non-redundant)
    non_redundant = {chosen_id: seq for seq, (chosen_id, _, _) in seq_index.items()}
    return non_redundant

def REMOVE_AMBIGUOUS(fasta_dict: Dict[str, str], cutoff: float = 0.1):

    ambiguous_bases = set("NRYWSKMBDHV")
    
    filtered_seqs = {}

    for id_str, seq in fasta_dict.items():
        seq_nogaps = seq.replace("-", "")
        amb_count = sum(base in ambiguous_bases for base in seq_nogaps)
        amb_count_p = amb_count/len(seq_nogaps)
        if amb_count_p >= cutoff:
            print(f"{id_str}: {amb_count} ambiguous bases ({amb_count_p:.2%})")
            continue
        else:
            filtered_seqs[id_str] = seq
    return filtered_seqs

def READ_FASTA(fasta_file: str,
               non_redundant: bool = False,
               remove_gaps: bool = False,
               remove_ambiguous: bool = True,
               cutoff_ambiguous: float = 0.1,
               date_format: str = "%Y-%m-%d",
               keep: str = "oldest") -> Dict[str, str]:
    """
    Read a FASTA file into a dict mapping sequence description -> sequence (uppercase).
    Optionally remove gaps and/or return a non-redundant dictionary.

    Parameters
    ----------
    fasta_file : str
        Path to the FASTA file to read.
    non_redundant : bool, optional
        If True, remove redundant identical sequences using remove_redundancy().
    remove_gaps : bool, optional
        If True, strip '-' characters from sequences.
    date_format : str, optional
        Date format passed to remove_redundancy when non_redundant=True.
    keep : {'oldest','newest'}, optional
        Policy to decide which ID to keep for duplicate sequences (passed to remove_redundancy).

    Returns
    -------
    Dict[str, str]
        Mapping from FASTA description -> processed sequence.
    """
    if fasta_file is None:
        raise ValueError("fasta_file path must be provided.")

    seq_dict: Dict[str, str] = {}

    for record in SeqIO.parse(fasta_file, "fasta"):
        id_str = str(record.description)
        seq = str(record.seq).upper().replace("*", "")

        if remove_gaps:
            seq = seq.replace("-", "")

        if len(seq) == 0:
            print(f"Warning: sequence for ID '{id_str}' has length 0 and will be skipped.")
            continue

        seq_dict[id_str] = seq
    
    if remove_ambiguous:
        seq_dict = REMOVE_AMBIGUOUS(fasta_dict=seq_dict, cutoff = cutoff_ambiguous)
    
    if non_redundant:

        return REMOVE_REDUNDANCY(fasta_dict=seq_dict,
                                 date_format=date_format,
                                 keep=keep)
    
    return seq_dict

def mer_split(sequence: Optional[str] = None,
              kmer_size: int = 6,
              overlap: int = 3, kmer_idx=False) -> List[str]:
    """
    Generate k-mer tokens of the form 'k{index}_{kmer}' from a sequence.

    Parameters
    ----------
    sequence : Optional[str]
        Input sequence. If None or empty, returns an empty list.
    kmer_size : int
        Length of each k-mer (must be > 0).
    overlap : int
        Number of characters overlapped between consecutive k-mers.

    Returns
    -------
    List[str]
        List of tokens like "k0_ATGCGA", "k1_TGCGAT", ...
    """
    if not sequence:
        return []

    seq = sequence.upper()
    L = len(seq)
    if kmer_size <= 0:
        raise ValueError("kmer_size must be > 0")
    if kmer_size > L:
        return []  # no complete k-mers

    step = max(1, kmer_size - overlap)
    tokens: List[str] = []
    idx = 0
    for i in range(0, L - kmer_size + 1, step):
        km = seq[i:i + kmer_size]
        if kmer_idx:
            tokens.append(f"k{idx}_{km}")
        else:
            tokens.append(f"{km}")
        idx += 1
    return tokens

def create_bag_of_mutations(
    fasta_dict: Dict[str, str], 
    reference_sequence: Optional[str] = None
) -> pd.DataFrame:
    """
    Creates a binary "Bag-of-Mutations" representation from a dictionary of aligned sequences.

    Each sequence is converted into a binary vector where each feature represents a specific 
    mutation relative to a reference sequence.

    Parameters
    ----------
    fasta_dict : Dict[str, str]
        A dictionary mapping sequence IDs to aligned sequences. All sequences must be of the same length.
    reference_sequence : Optional[str], optional
        The reference sequence to call mutations against. If None, the consensus sequence of the
        provided fasta_dict is calculated and used as the reference. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame where rows are sequence IDs, columns are mutations (e.g., 'A117G'),
        and values are 1 or 0, indicating the presence or absence of the mutation.
    
    Raises
    ------
    ValueError
        If the sequences in fasta_dict are not all of the same length.
    """
    if not fasta_dict:
        return pd.DataFrame()

    ids = list(fasta_dict.keys())
    sequences = list(fasta_dict.values())
    
    # Validate that all sequences have the same length
    seq_len = len(sequences[0])
    if not all(len(s) == seq_len for s in sequences):
        raise ValueError("All sequences in the alignment must have the same length.")

    # --- Step 1: Determine the reference sequence (if not provided) ---
    if reference_sequence is None:
        #logging.info("No reference sequence provided, calculating consensus...")
        consensus_chars = []
        for i in range(seq_len):
            #column = [seq[i] for seq in sequences if seq[i] != '-']
            column = [seq[i] for seq in sequences]
            if not column:
                consensus_chars.append('-')  # Column with only gaps
                continue
            # Find the most common character (mode) for the column
            most_common = Counter(column).most_common(1)[0][0]
            consensus_chars.append(most_common)
        reference_sequence = "".join(consensus_chars)
        print(reference_sequence)
        #logging.info("Consensus sequence calculated.")
    
    # --- Step 2: Identify all unique mutations (build the vocabulary) ---
    all_mutations = set()
    for seq in sequences:
        for i in range(seq_len):
            # A mutation is a difference from reference, but not a gap character in the query
            if reference_sequence[i] != seq[i]:
                mutation = f"{reference_sequence[i]}{i + 1}{seq[i]}" # Use 1-based indexing
                all_mutations.add(mutation)
    
    vocabulary = sorted(list(all_mutations))
    mutation_to_idx = {mutation: i for i, mutation in enumerate(vocabulary)}
    
    # --- Step 3: Create the binary matrix ---
    n_sequences = len(sequences)
    n_mutations = len(vocabulary)
    bom_matrix = np.zeros((n_sequences, n_mutations), dtype=np.uint8)

    for i, seq in enumerate(sequences):
        for j in range(seq_len):
            if reference_sequence[j] != seq[j] and seq[j] != '-':
                mutation = f"{reference_sequence[j]}{j + 1}{seq[j]}"
                if mutation in mutation_to_idx:
                    k = mutation_to_idx[mutation]
                    bom_matrix[i, k] = 1
    
    bom_df = pd.DataFrame(bom_matrix, index=ids, columns=vocabulary)
    return bom_df

def build_tfidf_vectorizer(
    fasta_dict: Dict[str, str],
    kmer_size: int = 6,
    overlap: int = 3,
    min_df: Union[int, float] = 1,
    max_df: Union[int, float] = 1.0,
    preserve_ids: bool = True,
    kmer_idx=False,
    return_sparse: bool = False,
    vectorizer_kwargs: Optional[Dict] = None,
    norm='l2'
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, TfidfVectorizer], Tuple]:
    """
    Build a TF-IDF representation from a dict mapping id -> sequence.

    Each sequence is tokenized into k-mers using `mer_split` and then joined into a
    single space-separated string that is passed to TfidfVectorizer with a simple
    split-tokenizer (pre-tokenized input).

    Parameters
    ----------
    fasta_dict : Dict[str, str]
        Mapping id -> sequence.
    kmer_size : int
        k-mer length.
    overlap : int
        overlap between consecutive k-mers.
    min_df, max_df : int/float
        Passed to TfidfVectorizer to control feature filtering.
    preserve_ids : bool
        If True, DataFrame index will be the original FASTA IDs; otherwise 0..N-1.
    return_sparse : bool
        If True, return the (sparse) TF-IDF matrix and the feature names instead of a dense DataFrame.
        Use this for large datasets to save memory.
    vectorizer_kwargs : Optional[Dict]
        Additional kwargs forwarded to TfidfVectorizer.

    Returns
    -------
    If return_sparse is False:
        pd.DataFrame: dense DataFrame (rows = sequences, columns = k-mer tokens).
    If return_sparse is True:
        tuple: (X_sparse, feature_names, ids) where
            - X_sparse is scipy.sparse matrix (n_samples x n_features),
            - feature_names is an array-like of token strings,
            - ids is the list of row identifiers (same order as rows in X_sparse).

    Notes
    -----
    - Converting to dense with `.toarray()` may blow memory on large datasets; prefer return_sparse=True.
    - We pass pre-tokenized strings and use a simple split tokenizer to preserve your exact tokens.
    """
    if fasta_dict is None:
        raise ValueError("fasta_dict must be provided (mapping id->sequence)")

    items = list(fasta_dict.items())
    if not items:
        warnings.warn("fasta_dict is empty -> returning empty DataFrame")
        return pd.DataFrame()

    ids = [k for k, _ in items] if preserve_ids else list(range(len(items)))
    sequences = [v for _, v in items]

    # Pre-tokenize
    list_mers = [mer_split(sequence=s, kmer_size=kmer_size, overlap=overlap, kmer_idx=kmer_idx) for s in sequences]
    corpus = [" ".join(tokens) for tokens in list_mers]

    empty_count = sum(1 for c in corpus if not c.strip())
    if empty_count == len(corpus):
        raise ValueError("All corpus are empty after k-mer extraction. "
                         "Check kmer_size/overlap vs sequence lengths.")
    if empty_count > 0:
        warnings.warn(f"{empty_count} documents are empty (no k-mers). They will be zero-rows in the TF-IDF matrix.")

    # default vectorizer args for pre-tokenized corpus
    v_kwargs = dict(norm=norm,
                    min_df=min_df,
                    max_df=max_df,
                    tokenizer=str.split,  # treat pre-joined tokens as space-separated
                    token_pattern=None,
                    preprocessor=None,
                    lowercase=False)
    if vectorizer_kwargs:
        v_kwargs.update(vectorizer_kwargs)

    vectorizer = TfidfVectorizer(**v_kwargs)
    X = vectorizer.fit_transform(corpus)  # sparse matrix

    feature_names = vectorizer.get_feature_names_out()

    if return_sparse:
        # return sparse matrix + feature names + ids (so user can reconstruct if needed)
        return X, feature_names, ids

    # convert to dense DataFrame (warning: may use a lot of memory)
    X_dense = X.toarray()
    df = pd.DataFrame(data=X_dense, index=ids, columns=feature_names)
    return df

def create_group_by(
    df: pd.DataFrame,
    Target_N: int = 100,
    date_col: str = "date",
    clade_col: str = "clade",
    date_freq: Optional[str] = "D",   # 'D' day, 'M' month, 'Y' year, 'Q' quarter, 'W' week, or None
    dropna: bool = True,
    min_group_size: int = 1,
    return_ordered: bool = True,
):
    """
    Group a DataFrame into (date_period -> clade -> DataFrame) and also return an
    ordered list of (date_period, clade, df_subset) for easy iteration.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least the date and clade columns.
    date_col : str
        Name of the date column (will be converted to datetime if necessary).
    clade_col : str
        Name of the clade (grouping) column.
    date_freq : Optional[str]
        Frequency key for bucketing dates:
          - 'D' = day (default)
          - 'M' = month
          - 'Y' = year
          - 'Q' = quarter
          - 'W' = week
          - None = use exact date values (no bucketing)
    dropna : bool
        If True, drop rows with missing values in date_col or clade_col.
    min_group_size : int
        Discard groups with fewer than this many rows (default: 1, i.e., keep all).
    return_ordered : bool
        If True, the nested dictionary is an OrderedDict (sorted by date then clade).
        If False, a plain dict is returned (insertion order preserved in modern Python
        but not explicitly sorted).

    Returns
    -------
    nested_dict, list_of_tuples
        nested_dict: {date_period_timestamp: {clade_value: df_subset, ...}, ...}
        list_of_tuples: [(date_period_timestamp, clade_value, df_subset), ...]
        The list is ordered by date_period then clade.
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")
    if clade_col not in df.columns:
        raise KeyError(f"Clade column '{clade_col}' not found in DataFrame.")
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")

    # Work on a copy to avoid modifying original
    df2 = df.copy()

    # Ensure date column is datetime (coerce invalid values to NaT)
    if not pd.api.types.is_datetime64_any_dtype(df2[date_col]):
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")

    if dropna:
        df2 = df2.dropna(subset=[date_col, clade_col])

    # Build date period column according to requested frequency
    if date_freq is None:
        df2["_date_period"] = df2[date_col]
    else:
        freq = date_freq.upper()
        if freq not in {"D", "M", "Y", "Q", "W"}:
            raise ValueError("date_freq must be one of {'D','M','Y','Q','W'} or None")
        if freq == "D":
            df2["_date_period"] = df2[date_col].dt.floor("D")
        elif freq == "M":
            df2["_date_period"] = df2[date_col].dt.to_period("M").dt.to_timestamp()
        elif freq == "Y":
            df2["_date_period"] = df2[date_col].dt.to_period("Y").dt.to_timestamp()
        elif freq == "Q":
            df2["_date_period"] = df2[date_col].dt.to_period("Q").dt.to_timestamp()
        elif freq == "W":
            df2["_date_period"] = df2[date_col].dt.to_period("W").dt.to_timestamp()

    # Sort so grouping order is deterministic
    df2 = df2.sort_values(["_date_period", clade_col])

    nested: Dict[pd.Timestamp, Dict[str, pd.DataFrame]] = OrderedDict() if return_ordered else {}
    list_out: List[Tuple[pd.Timestamp, str, pd.DataFrame]] = []

    for date_period, df_date in df2.groupby("_date_period"):
        
        clade_counts = df_date['clade'].value_counts()
        #print(N_clades)
        N_clades = len(clade_counts)
        W = clade_counts/len(df_date)
        #print(f"valor: {W}")
        clade_alloc = (W * (Target_N - N_clades)).round() + 1
        print(f"Target: {Target_N} ---> Alloc: {clade_alloc} ")

        inner: Dict[str, pd.DataFrame] = OrderedDict() if return_ordered else {}

        for clade_val, df_clade in df_date.groupby(clade_col):
            if len(df_clade) < min_group_size:
                continue
            df_slice = df_clade.drop(columns=["_date_period"])
            df_slice.reset_index(inplace=True, drop=True)
            inner[clade_val] = df_slice
            list_out.append((date_period, clade_val, df_slice, clade_alloc[clade_val]))
        if inner:
            nested[date_period] = inner
    
    return nested, list_out

def create_TFIDF_groups(list_of_groups: list = None, vector_dataframe: pd.DataFrame = None) -> List[pd.DataFrame]:
    """
    Parameters
    ----------
    list_of_groups : list
        The list of group tuples generated by `create_group_by`.
    vector_dataframe : pd.DataFrame
        The global DataFrame (sparse or dense) containing vectors for all sequences.

    Returns
    -------
    List[pd.DataFrame]
        A list of DataFrames, where each DataFrame contains the vectors for a group.
    """
    
    # Verificação para garantir que o índice está otimizado para busca.
    # Se o índice não for único, a busca com .loc pode ser lenta.
    if not vector_dataframe.index.is_unique:
        warnings.warn("The vector DataFrame index is not unique. Performance of .loc may be degraded.")

    list_of_groups_vectors = []
    
    for group_info in tqdm(list_of_groups, desc="Extracting group vectors"):
        # group_info is a tuple: (date, clade, metadata_df, alloc)
        group_metadata_df = group_info[2]

        names = group_metadata_df['name'].values

        group_vectors_df = vector_dataframe.loc[names]
        
        list_of_groups_vectors.append(group_vectors_df)

    return list_of_groups_vectors

# -----------------------
# util: Create distance Matrix (cosine is recomended to TF-IDF)
# -----------------------
def build_distance_matrix_from_tfidf(X, metric: str = "cosine", reduce_dim: Optional[int] = None, random_state: int = 42):
    """
    X : pd.DataFrame (dense) or scipy.sparse / ndarray
    metric: passed to sklearn.metrics.pairwise_distances
    reduce_dim: optional, int -> run TruncatedSVD to reduce before computing distances
    Returns: distance_matrix (n x n), index_list (same order as X)
    """
    is_df = isinstance(X, pd.DataFrame)
    if is_df:
        idx = X.index.to_list()
        Xmat = X.values
    else:
        Xmat = X
        idx = list(range(Xmat.shape[0]))

    if reduce_dim is not None:
        
        n_comp = min(reduce_dim, Xmat.shape[1], Xmat.shape[0] - 1)
        if n_comp <= 0:
            embedding = Xmat
        else:
            svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
            embedding = svd.fit_transform(Xmat)
    else:
        embedding = Xmat

    D = pairwise_distances(embedding, metric=metric, n_jobs=-1)
    # ensure numeric stability
    np.fill_diagonal(D, 0.0)
    return D, idx

# -----------------------
# objective functions given distance matrix
# -----------------------
def diversity_score_from_indices(D: np.ndarray, indices: list, objective: str = "avg", eps: float = 1e-8):
    """
    D: matriz de distâncias (n x n)
    indices: lista/int array dos índices selecionados
    objective: 'avg' | 'min' | 'sumlogdet'
    """
    if len(indices) <= 1:
        return 0.0
    sub = D[np.ix_(indices, indices)]
    n = sub.shape[0]
    if objective == "avg":
        triu_idx = np.triu_indices(n, k=1)
        return float(np.mean(sub[triu_idx]))
    elif objective == "min":
        triu_idx = np.triu_indices(n, k=1)
        return float(np.min(sub[triu_idx]))
    elif objective == "sumlogdet":
        # similaridade -> K = 1 - distance (clamp)
        K = 1.0 - sub
        K = (K + K.T) / 2.0
        K = K + eps * np.eye(n)
        sign, logdet = slogdet(K)
        if sign <= 0:
            return -1e9
        return float(logdet)
    else:
        raise ValueError("objective must be one of {'avg','min','sumlogdet'}")

# -----------------------
# Genetic Algorithm for subset selection
# -----------------------

def genetic_select_subset(
    D: np.ndarray,
    k: int,
    objective: str = "min",
    pop_size: int = 200,
    generations: int = 200,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.25,
    tournament_size: int = 3,
    elitism: int = 2,
    random_state: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Genetic algorithm that evolves populations of subsets (lists of k unique indices).
    Returns dict with best_indexes, best_score, history etc.
    """
    rng = random.Random(random_state)
    np.random.seed(random_state)
    n = D.shape[0]
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and n")

    # initialize population: random subsets
    def rand_subset():
        return list(np.random.choice(n, size=k, replace=False))

    pop = [rand_subset() for _ in range(pop_size)]

    # evaluate
    def fitness(ind):
        return diversity_score_from_indices(D, ind, objective=objective)

    scores = [fitness(ind) for ind in pop]
    best_idx = int(np.argmax(scores))
    best = pop[best_idx]
    best_score = scores[best_idx]
    history = [best_score]

    def tournament_select():
        # return copy of selected individual's subset
        aspirants = rng.sample(range(pop_size), tournament_size)
        best_a = max(aspirants, key=lambda i: scores[i])
        return list(pop[best_a])

    def crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        # union-based crossover: preserve intersection, fill rest by random from union/diff
        set_a, set_b = set(a), set(b)
        inter = list(set_a & set_b)
        union = list(set_a | set_b)
        # offspring1: start with intersection, then sample from union\intersection till k
        def make_child():
            child = list(inter)
            needed = k - len(child)
            # pick from symmetric difference first to preserve parent traits
            diff = [x for x in union if x not in child]
            if needed > 0:
                take = rng.sample(diff, k=needed)
                child.extend(take)
            rng.shuffle(child)
            return child
        return make_child(), make_child()

    def mutate(ind: List[int]) -> List[int]:
        ind_set = set(ind)
        # with some probability perform a swap: replace 1..m positions
        if rng.random() < mutation_rate:
            # how many swaps
            m = rng.randint(1, max(1, int(0.25 * k)))
            for _ in range(m):
                # choose an element to remove and one to add
                remove = rng.choice(ind)
                candidates = [x for x in range(n) if x not in ind_set]
                if not candidates:
                    break
                add = rng.choice(candidates)
                ind.remove(remove)
                ind.append(add)
                ind_set.remove(remove); ind_set.add(add)
        return ind

    for gen in range(generations):
        new_pop = []
        # elitism: carry top E
        sorted_idx = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        for e in range(elitism):
            new_pop.append(list(pop[sorted_idx[e]]))

        while len(new_pop) < pop_size:
            parent1 = tournament_select()
            parent2 = tournament_select()
            if rng.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)

        pop = new_pop
        scores = [fitness(ind) for ind in pop]
        gen_best_idx = int(np.argmax(scores))
        gen_best_score = scores[gen_best_idx]
        if gen_best_score > best_score:
            best_score = gen_best_score
            best = list(pop[gen_best_idx])
        history.append(best_score)
        if verbose and gen % max(1, generations // 10) == 0:
            print(f"gen {gen}/{generations} best={best_score:.6f}")

    return {"best_indexes": sorted(best), "best_score": best_score, "history": history}

# -----------------------
# Differential Evolution (continuous vector => top-k by ranking)
# -----------------------
def de_select_subset(
    D: np.ndarray,
    k: int,
    objective: str = "avg",
    pop_size: int = 50,
    generations: int = 200,
    F: float = 0.8,
    CR: float = 0.9,
    random_state: int = 42,
    verbose: bool = False
) -> Dict:
    """
    DE-like algorithm over continuous vectors v in [0,1]^n.
    Each vector maps to subset = argsort(-v)[:k] (top-k).
    Uses classic DE/rand/1/bin mutation + greedy selection.
    """
    rng = np.random.RandomState(random_state)
    n = D.shape[0]
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and n")

    # initialize population: uniform random [0,1]
    pop = rng.rand(pop_size, n)
    # evaluate
    def vec_to_indices(v):
        return np.argsort(-v)[:k]
    def fitness_vec(v):
        inds = vec_to_indices(v)
        return diversity_score_from_indices(D, inds, objective=objective)

    fitness = np.array([fitness_vec(pop[i]) for i in range(pop_size)])
    best_idx = int(np.argmax(fitness))
    best_v = pop[best_idx].copy()
    best_score = float(fitness[best_idx])
    history = [best_score]

    for gen in range(generations):
        for i in range(pop_size):
            # choose a,b,c distinct and distinct from i
            idxs = [i]
            while len(idxs) < 4:
                cand = rng.randint(0, pop_size)
                if cand not in idxs:
                    idxs.append(cand)
            a, b, c = pop[idxs[1]], pop[idxs[2]], pop[idxs[3]]
            mutant = a + F * (b - c)
            # clip
            mutant = np.clip(mutant, 0.0, 1.0)
            # crossover binomial
            cross = rng.rand(n) < CR
            if not np.any(cross):
                cross[rng.randint(0, n)] = True
            trial = np.where(cross, mutant, pop[i])
            # selection
            f_trial = fitness_vec(trial)
            if f_trial > fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial > best_score:
                    best_score = f_trial
                    best_v = trial.copy()
        history.append(best_score)
        if verbose and gen % max(1, generations // 10) == 0:
            print(f"DE gen {gen}/{generations} best={best_score:.6f}")

    best_indexes = np.argsort(-best_v)[:k].tolist()
    return {"best_indexes": sorted(best_indexes), "best_score": best_score, "history": history}

# -----------------------
# high-level wrapper
# -----------------------
def maximize_diversity_indices(
    X,                    # tfidf DataFrame or matrix
    k: int = 10,
    method: str = "ga",   # "ga" or "de"
    objective: str = "min",
    precompute_metric: str = "cosine",
    reduce_dim_before_distance: Optional[int] = None,
    random_state: int = 42,
    **kwargs
) -> Dict:
    """
    top-level helper: builds distance matrix and runs chosen optimizer.
    kwargs forwarded to genetic_select_subset or de_select_subset.
    Returns dict with best indices (integers) and optionally index labels if X is DataFrame.
    """
    D, idx = build_distance_matrix_from_tfidf(X, metric=precompute_metric, reduce_dim=reduce_dim_before_distance, random_state=random_state)
    if method == "ga":
        res = genetic_select_subset(D, k=k, objective=objective, random_state=random_state, **kwargs)
    elif method == "de":
        res = de_select_subset(D, k=k, objective=objective, random_state=random_state, **kwargs)
    else:
        raise ValueError("method must be 'ga' or 'de'")
    # attach original ids if X was DataFrame
    if isinstance(X, pd.DataFrame):
        res["best_ids"] = [idx[i] for i in res["best_indexes"]]
    else:
        res["best_ids"] = res["best_indexes"]
    return res

def dedup_by_region_sequence_keep_oldest(meta_df,
                                          seq_col='sequence',
                                          region_col='region',
                                          date_col='date',
                                          normalize_seq=True):
    """
    Remove duplicates by (region, sequence), keeping the oldest sample per group.
    Returns: meta_kept (reset index), dropped_original_indices (list)
    """

    df = meta_df.copy()

    # preserve original integer row positions
    df = df.reset_index().rename(columns={'index': 'orig_index'})

    # parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # normalize sequences if requested
    if normalize_seq:
        df['_seq_norm'] = df[seq_col].astype(str).str.strip().str.upper()

    else:
        df['_seq_norm'] = df[seq_col].astype(str)

    # treat missing dates as very recent so they are NOT chosen as oldest
    df['_date_filled'] = df[date_col].fillna(pd.Timestamp.max)

    # sort so oldest (smallest date) and lowest orig_index come first within each group
    df = df.sort_values([region_col, '_seq_norm', '_date_filled', 'orig_index'],
                        ascending=[True, True, True, True])

    # drop duplicates keeping the first o   ccurrence (the oldest + tie-break by orig_index)
    kept = df.drop_duplicates(subset=[region_col, '_seq_norm'], keep='first').copy()

    # dropped original integer indices
    all_orig = set(df['orig_index'].tolist())
    kept_orig = set(kept['orig_index'].tolist())
    dropped_orig_indices = sorted(list(all_orig - kept_orig))

    # cleanup and return meta kept (reset index)
    kept = kept.drop(columns=['_seq_norm', '_date_filled', 'orig_index'])
    kept = kept.reset_index(drop=True)

    return kept, dropped_orig_indices

def split_tfidf_by_region(meta_df, tfidf_df, region_col='region'):
    """
    Retorna um dict: region -> tfidf_subdf (preserva índices originais).
    meta_df and tfidf_df must have the same number of rows and aligned order/index.
    """
    if len(meta_df) != len(tfidf_df):
        raise ValueError("meta_df and tfidf_df must have same number of rows")
    # ensure aligned index
    tfidf_df = tfidf_df.reset_index(drop=True)
    meta = meta_df.reset_index(drop=True)
    groups_tfidf = {}
    groups_meta = {}
    for region, idxs in meta.groupby(region_col).groups.items():
        groups_tfidf[region] = tfidf_df.loc[list(idxs)].copy()
        groups_meta[region] = meta.loc[list(idxs)].copy()
    return groups_meta, groups_tfidf

def region_counts(meta_df, region_col='region'):
    """Return dict region -> count"""
    return meta_df[region_col].value_counts().to_dict()

def compute_diversity_score_per_region(tfidf_by_region):
    """
    Quick diversity proxy per region:
    Use mean feature variance (across columns) normalized.
    tfidf_by_region: dict region -> tfidf_subdf
    returns dict region->div_norm in [0,1]
    """
    div = {}
    for c, df in tfidf_by_region.items():
        if df.shape[0] <= 1:
            div[c] = 0.0
            continue
        # compute variance per column then mean (works on dense DataFrame)
        # if df is sparse, convert to dense for variance on manageable sizes
        try:
            variances = df.var(axis=0, ddof=0)  # Series of variances
            score = variances.mean()
        except Exception:
            # fallback: convert to numpy
            arr = df.to_numpy()
            score = np.var(arr, axis=0).mean() if arr.shape[1] > 0 else 0.0
        div[c] = float(score)
    # normalize to 0..1
    vals = np.array(list(div.values()))
    if vals.max() == vals.min():
        return {k: 0.0 for k in div}
    vmin = vals.min(); vmax = vals.max()
    div_norm = {k: (v - vmin) / (vmax - vmin) for k, v in div.items()}
    return div_norm

def allocate_proportional(meta_df=None, tfidf_df=None, counts=None,
                                    target_N=100,
                                    region_col='region',
                                    alpha=0.5,
                                    preserve_max_share=0.8,
                                    use_diversity=False,
                                    beta=0.2):
    """
    Aloca amostras garantindo pelo menos 1 por país (quando possível), respeitando preserve_max_share,
    e então aloca o restante proporcionalmente aos pesos (counts ** alpha) ajustados por diversidade se pedida.

    Parâmetros principais:
      - meta_df / tfidf_df: usado se counts não for fornecido.
      - counts: dict region->count (se fornecido, ignora meta_df/tfidf_df).
      - target_N: número total de amostras desejado.
      - alpha: expoente para suavizar proporcionalidade (0..1).
      - preserve_max_share: fração do target_N que pode ser consumida pela garantia "1 por país".
      - use_diversity / beta: se True, usa scores de diversidade para ajustar pesos.
    Retorna:
      - alloc: dict region->n_keep (total = min(target_N, total_available)).
    """
    # --- obter counts ---
    if counts is None:
        if meta_df is None:
            raise ValueError("Forneça counts ou meta_df")
        counts = meta_df[region_col].value_counts().to_dict()
    # filtrar países com zero
    counts = {c: int(cnt) for c, cnt in counts.items() if cnt > 0}
    countries = sorted(counts.keys())

    total_avail = sum(counts.values())
    if target_N >= total_avail:
        return dict(counts)

    # inicializa alloc com 0 para todos (evita KeyError)
    alloc = {c: 0 for c in countries}

    # --- preparar diversidade se necessário ---
    if use_diversity and meta_df is not None and tfidf_df is not None:
        _, tfidf_by_region = split_tfidf_by_region(meta_df, tfidf_df, region_col=region_col)
        div_norm = compute_diversity_score_per_region(tfidf_by_region)
    else:
        div_norm = {c: 0.0 for c in countries}

    # --- etapa 1: garantir 1 por país (quando possível) ---
    desired_one_per = [c for c in countries if counts[c] >= 1]
    desired_reserved = len(desired_one_per)
    preserve_budget = int(math.floor(max(0.0, min(1.0, preserve_max_share)) * target_N))

    if desired_reserved <= preserve_budget:
        # podemos dar 1 para todos os países
        for c in desired_one_per:
            alloc[c] = 1
        reserved = desired_reserved
    else:
        # não há budget para todos; priorizamos países com menor disponibilidade (counts pequenos)
        # alternativa: priorizar por outra métrica (ex.: importância), aqui usamos counts asc
        ordered = sorted(desired_one_per, key=lambda x: counts[x])  # os menores primeiro
        for i, c in enumerate(ordered):
            if i < preserve_budget:
                alloc[c] = 1
            else:
                alloc[c] = 0
        reserved = sum(alloc.values())

    remaining_slots = target_N - reserved
    if remaining_slots <= 0:
        # já consumimos o target com os "1 por país" (ou o cap impôs)
        # se houver países com alloc > counts (improvável aqui), corrige:
        for c in alloc:
            alloc[c] = min(alloc[c], counts[c])
        return alloc

    # --- etapa 2: distribuir o restante proporcionalmente ---
    # construir pesos para todas as countries que ainda não atingiram seu limite
    weights = {}
    candidates = [c for c in countries if alloc[c] < counts[c]]
    for c in candidates:
        w = (counts[c] ** alpha)
        w = w * (1.0 + beta * float(div_norm.get(c, 0.0)))
        weights[c] = max(0.0, float(w))

    W = sum(weights.values())
    if W <= 0:
        # fallback: distribuir uniformemente entre candidatos
        base = remaining_slots // len(candidates) if candidates else 0
        for c in candidates:
            alloc[c] += min(counts[c] - alloc[c], base)
        left = remaining_slots - sum(min(counts[c] - alloc[c], base) for c in candidates)
        order = sorted(candidates, key=lambda x: counts[x], reverse=True)
        i = 0
        while left > 0 and order:
            c = order[i % len(order)]
            if alloc[c] < counts[c]:
                alloc[c] += 1
                left -= 1
            i += 1
        return alloc

    # calcular quotas reais (floors) e restos para distribuir o leftover de forma justa
    raw_shares = {c: remaining_slots * (weights[c] / W) for c in candidates}
    floored = {c: int(math.floor(raw_shares[c])) for c in candidates}
    # aplicar floor (respeitando máximo counts)
    for c in candidates:
        add = min(floored[c], counts[c] - alloc[c])
        alloc[c] += add

    assigned = sum(floored.values())
    leftover = remaining_slots - sum(min(floored[c], counts[c] - (alloc[c]-min(floored[c], counts[c] - (alloc[c]-floored[c])))) for c in candidates)
    # simpler compute: recompute leftover as difference to be safe
    leftover = target_N - sum(alloc.values())

    # distribuir leftover pelos maiores restos fracionários, priorizando países com mais capacidade
    if leftover > 0:
        frac_remainders = [(c, raw_shares[c] - math.floor(raw_shares[c])) for c in candidates]
        frac_remainders.sort(key=lambda x: (x[1], weights[x[0]]), reverse=True)
        i = 0
        while leftover > 0 and frac_remainders:
            c, _ = frac_remainders[i % len(frac_remainders)]
            if alloc[c] < counts[c]:
                alloc[c] += 1
                leftover -= 1
            i += 1
            # to avoid infinite loop, break if iterated too many times (shouldn't happen)
            if i > len(frac_remainders) * 5:
                break

    # --- segurança final: não exceder counts e somar exatamente target_N ---
    for c in alloc:
        if alloc[c] > counts[c]:
            alloc[c] = counts[c]

    total_alloc = sum(alloc.values())
    if total_alloc != target_N:
        diff = target_N - total_alloc
        if diff > 0:
            # adicionar a países com capacidade, por peso
            capacity = [c for c in countries if alloc[c] < counts[c]]
            capacity.sort(key=lambda x: weights.get(x, 0), reverse=True)
            i = 0
            while diff > 0 and capacity:
                c = capacity[i % len(capacity)]
                alloc[c] += 1
                diff -= 1
                i += 1
        elif diff < 0:
            # remover de países com maior alocação (>1), preferir os que não eram dos "1 garantidos"
            diff = -diff
            reducibles = sorted([c for c in countries if alloc[c] > 1], key=lambda x: alloc[x], reverse=True)
            i = 0
            while diff > 0 and reducibles:
                c = reducibles[i % len(reducibles)]
                if alloc[c] > 1:
                    alloc[c] -= 1
                    diff -= 1
                i += 1

    return alloc

def adaptive_pop_size(n, k, min_pop=10, max_pop=2000):
    """
    Escolhe pop_size adaptativo dado n e k.
    Se o número de combinações for pequeno, reduz pop_size para <= combinações.
    """
    total_combinations = comb(n, k)

    if total_combinations <= 10000:
        # pode explorar tudo sem GA, mas se quiser GA: pop = min(total_combinations, max_pop)
        return min(total_combinations, max_pop)
    
    # heurística padrão se o espaço é grande
    heuristic = int(20 * math.log(max(2, n)) * math.sqrt(max(1, k)))
    pop_size = min(max_pop, max(min_pop, heuristic))
    return pop_size

def run_hybrid_sampling(alloc_values: int, metadata_subgroup: pd.DataFrame, random_state=42) -> List[int]:
    """
    Seleciona um subconjunto de índices de um DataFrame de metadados usando a abordagem híbrida.

    1. Mantém todas as sequências únicas (singletons).
    2. Preenche os slots restantes com uma amostra aleatória das sequências comuns.
    
    Retorna:
        Uma lista dos índices originais do DataFrame a serem mantidos.
    """
    # Se o grupo já é pequeno o suficiente, retorna todos os seus índices
    if len(metadata_subgroup) <= alloc_values:
        return metadata_subgroup.index.to_list()

    # Identifica sequências únicas (singletons) e comuns
    seq_counts = metadata_subgroup['sequence'].value_counts()
    singletons_seqs = seq_counts[seq_counts == 1].index
    common_seqs = seq_counts[seq_counts > 1].index

    df_singletons = metadata_subgroup[metadata_subgroup['sequence'].isin(singletons_seqs)]
    df_common = metadata_subgroup[metadata_subgroup['sequence'].isin(common_seqs)]

    # Se os singletons já preenchem ou excedem a alocação, amostra aleatoriamente deles
    if len(df_singletons) >= alloc_values:
        return df_singletons.sample(n=alloc_values, random_state=random_state).index.to_list()
    
    # Caso principal: Pega todos os singletons e preenche o resto com os comuns
    else:
        num_common_to_sample = alloc_values - len(df_singletons)
        
        # Pega os índices dos singletons
        singleton_indices = df_singletons.index.to_list()
        
        # Amostra aleatoriamente os índices dos comuns
        common_indices_sample = df_common.sample(n=num_common_to_sample, random_state=random_state).index.to_list()
        
        # Retorna a lista combinada de índices
        return singleton_indices + common_indices_sample

def run_filter_sequences(alloc_values: int = None, tfidf_dataframe: pd.DataFrame = None, 
            min_pop: int =10, max_pop: int =2000, precompute_metric="cosine",
            reduce_dim_before_distance=None,
            generations: int =200,
            tournament_size: int =3,
            elitism: int =2,
            mutation_rate: float =0.8,
            random_state=42,
            verbose=True,
            crossover_rate: float =0.9):
    
    if len(tfidf_dataframe) <= alloc_values:
    
        index_list = tfidf_dataframe.index.to_list()
    
        return index_list
    
    else:
    
        # parameters
        k = alloc_values
        pop_size = adaptive_pop_size(n=len(tfidf_dataframe), k=k, min_pop=min_pop, max_pop=max_pop)

        # Running GA
        res_ga = maximize_diversity_indices(
            tfidf_dataframe,                      
            k=k,
            method="ga",                    
            objective="min",                
            precompute_metric=precompute_metric,     
            reduce_dim_before_distance=reduce_dim_before_distance,
            pop_size=pop_size,                   
            generations=generations,             
            tournament_size=tournament_size,
            elitism=elitism,
            mutation_rate=mutation_rate,
            random_state=random_state,
            verbose=verbose,
            crossover_rate=crossover_rate
        )
        
        index_list = res_ga["best_ids"]
        
        return index_list

def main():
    parser = argparse.ArgumentParser(description="GenoSieve: A subsampling tool designed to maximize diversity while preserving representativeness. " \
    "Developed by Vinicius Carius de Souza.", \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- Definições de Argumentos ---
    parser.add_argument("--FASTA_ALN", required=True, type=str, help="Sequence aligment file in FASTA format")
    parser.add_argument("--METADATA", required=True, type=str, help="TSV metadata file that includes required columns 'name', 'date', 'region', and 'clade'")
    parser.add_argument("--output", required=False, type=str, help="Output file in FASTA format", default="subsampling.fasta")
    parser.add_argument("--keep_header", required=False, action="store_true", help="It will keep the fasta header as the same.")
    parser.add_argument("--color_map", default="nipy_spectral", required=False, help="Check option in 'https://matplotlib.org/stable/users/explain/colors/colormaps.html'.")
    parser.add_argument("--cutoff_ambiguous", default=0.1, type=float, required=False, help="Cutoff of ambiguous bases permited. Accept values from 0 up to 1.")
    # Grupo de argumentos para a estratégia de subsampling
    sampling_group = parser.add_argument_group('Sampling Strategy')
    sampling_group.add_argument("--dedup", required=False, action="store_true", help="Remove duplicate sequences from subgroups considering the same period, clade, and region. " \
                                                                            "The dedup command will keep the oldest sequence from each subgroup.")
    sampling_group.add_argument("--date_freq", required=False, type=str, help="Frequency to group sequences by time.", default="M")
    sampling_group.add_argument("--target_N", required=True, type=int, help="Number of samples to allocate PER time period (defined by --date_freq).")
    #sampling_group.add_argument("--maintain_clade_proportions", required=False, action="store_true", help="If set, target_N is distributed among clades to strictly maintain original proportions. " \
    #                                                                                                    "Default behavior smooths proportions to better represent smaller clades.")
    
    # Grupo de argumentos para a alocação
    allocation_group = parser.add_argument_group('Allocation Parameters')
    allocation_group.add_argument("--alpha", required=False, type=float, help="Exponent to smooth proportionality (0.0 to 1.0). Lower values give more weight to smaller groups.", default=0.5)
    allocation_group.add_argument("--preserve_max_share", required=False, type=float, help="Fraction of target_N that can be consumed by the '1 per region' guarantee.", default=0.8)
    allocation_group.add_argument("--use_diversity", required=False, action="store_true", help="Use genetic diversity score to adjust allocation weights.")
    allocation_group.add_argument("--beta", required=False, type=float, help="Weight for the diversity score if --use_diversity is active.", default=0.2)

    # Grupo de argumentos para a vetorização
    vectorizer_group = parser.add_argument_group('Vectorizer Parameters')
    vectorizer_group.add_argument("--kmer_size", required=False, type=int, help="K-mer size for TF-IDF vectorization.", default=6)
    vectorizer_group.add_argument("--overlap", required=False, type=int, help="Overlap between k-mers.", default=3)
    vectorizer_group.add_argument("--kmer_idx", required=False, action="store_true", help="It will create a positional k-mer information.")
    vectorizer_group.add_argument("--use_TFIDF", required=False, action="store_true", help="Vectorizer sequences using TF-IDF method from k-mers.")
    vectorizer_group.add_argument("--reference_name", required=False, type=str, help="It will be used only when not applied `use_TFIDF`.", default=None)

    # Grupo de argumentos para o Algoritmo Genético
    ga_group = parser.add_argument_group('Genetic Algorithm Parameters')
    ga_group.add_argument("--use_GA", required=False, action="store_true", help="Use a genetic algorithm to maximize diversity within groups. Default is Hybrid Sampling.")
    ga_group.add_argument("--objective_function", required=False, choices=['avg', 'min', 'sumlogdet'], help="Fitness function for the GA to maximize.", default="min")
    ga_group.add_argument("--min_pop", required=False, type=int, help="Minimum population size for the GA.", default=10)
    ga_group.add_argument("--max_pop", required=False, type=int, help="Maximum population size for the GA.", default=2000)
    ga_group.add_argument("--generations", required=False, type=int, help="Number of generations for the GA.", default=200)
    ga_group.add_argument("--mutation_rate", required=False, type=float, help="Probability of mutation in the GA.", default=0.25)   
    ga_group.add_argument("--crossover_rate", required=False, type=float, help="Probability of crossover in the GA.", default=0.9)
    ga_group.add_argument("--tournament_size", required=False, type=int, help="Tournament size for selection in the GA.", default=3)
    ga_group.add_argument("--elitism", required=False, type=int, help="Number of elite individuals to carry to the next generation.", default=2)

    # Grupo de argumentos para o cálculo da distância
    distance_group = parser.add_argument_group('Distance Calculation Parameters')
    distance_group.add_argument("--precompute_metric", required=False, type=str, help="Distance metric for TF-IDF vectors (e.g., 'cosine', 'euclidean').", default="cosine")
    distance_group.add_argument("--reduce_dim_before_distance", required=False, type=int, help="Target dimension for TruncatedSVD before distance calculation.", default=None)
    
    # Grupo de argumentos de controle
    control_group = parser.add_argument_group('Control Parameters')
    control_group.add_argument("--random_seed", required=False, type=int, help="Seed for random number generators to ensure reproducibility.", default=1991)
    control_group.add_argument("--verbose", required=False, action="store_true", help="Enable verbose logging output to console and file.")

    args = parser.parse_args()

    # --- Configuração do Logging ---
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("GenoSieve_running.log", mode='w'),
                            logging.StreamHandler()
                        ])
    
    logging.info("GenoSieve Subsampling Tool Started")
    logging.info("Command line arguments:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  - {arg}: {value}")
    
    if args.dedup and not args.use_GA:
        logging.warning(
            "Using --dedup without --use_GA is not recommended for phylodynamic analysis, "
            "as it removes frequency signals before the hybrid sampling is applied."
        )

    # --- Carregamento e Preparação dos Dados ---
    logging.info(f"Reading metadata from {args.METADATA}...")
    METADATA = READ_METADATA(args.METADATA)
    logging.info(f"Reading sequences from {args.FASTA_ALN}...")
    ALN = READ_FASTA(fasta_file=args.FASTA_ALN, remove_ambiguous=True, cutoff_ambiguous=args.cutoff_ambiguous)

    logging.info("Merging metadata and sequences...")
    METADATA['sequence'] = METADATA['name'].map(ALN)
    
    initial_count = len(METADATA)
    METADATA.dropna(subset=["sequence"], inplace=True)
    METADATA.reset_index(inplace=True, drop=True)
    
    logging.info(f"Initial sample count: {initial_count}. After removing samples not in FASTA: {len(METADATA)}.")

    logging.info(f"Defining clade colors according to {args.color_map} color map...")

    # Extract unique clades present in the DataFrame
    unique_clades = set(METADATA['clade'])

    assert args.color_map in list(colormaps), f"{args.color_map} was not found in matplotlib colomaps. Check option in 'https://matplotlib.org/stable/users/explain/colors/colormaps.html'."

    # Create a dictionary mapping each clade to a unique color using the Paired colormap
    cmap = plt.get_cmap(args.color_map)
    n = len(unique_clades)
    clade_color_dict = {clade: cmap(i / max(1, n-1)) for i, clade in enumerate(unique_clades)}
    
    #clade_color_dict = {clade: plt.get_cmap(args.color_map).colors[i] for i, clade in enumerate(unique_clades)}

    logging.info("Calculating the clade frequence by dates for original Dataset...")
    
    original_freq_by_month = streamPlot(df=METADATA, clade_color_dict=clade_color_dict, output="original_clade_frequencies_by_date.pdf" , figsize=(24, 8))
    
    logging.info(f"{original_freq_by_month.mean(axis=1)}")

    # --- Agrupamento e Deduplicação (se solicitado) ---
    logging.info(f"Creating initial groups by date (frequency: {args.date_freq}) and clade...")
    _, list_of_groups = create_group_by(METADATA, Target_N=args.target_N, date_col='date', clade_col='clade', date_freq=args.date_freq)
    logging.info(f"Created {len(list_of_groups)} initial groups.")

    if args.dedup:
        logging.info("Removing duplicated sequences by region for each group...")
        total_before = len(METADATA)
        
        # O processo de dedup junta os dataframes dos grupos, então precisamos de um novo METADATA
        TMP_list = []
        for _, _, group_df in tqdm(list_of_groups, desc="Deduplicating groups"):
            meta_clean, _ = dedup_by_region_sequence_keep_oldest(group_df, seq_col='sequence', date_col='date')
            TMP_list.append(meta_clean)
        
        METADATA = pd.concat(TMP_list, ignore_index=True)
        METADATA.reset_index(inplace=True, drop=True)
        logging.info(f"Deduplication finished. Count before: {total_before}, after: {len(METADATA)}.")
        
        logging.info("Recreating groups after deduplication...")
        _, list_of_groups = create_group_by(METADATA, Target_N=args.target_N, date_col='date', clade_col='clade', date_freq=args.date_freq)
        logging.info(f"Recreated {len(list_of_groups)} groups.")

    # --- Vetorização TF-IDF ---
    if args.use_TFIDF:
        logging.info("Building TF-IDF vectorizer for all sequences...")
        # Usamos o dicionário ALN original para construir o vetorizador para garantir consistência
        TFIDF_VEC = build_tfidf_vectorizer(kmer_size=args.kmer_size, overlap=args.overlap, fasta_dict=ALN, norm='l2', kmer_idx=args.kmer_idx)
        logging.info("Building TF-IDF representations for each group...")
        list_of_groups_tfidf = create_TFIDF_groups(list_of_groups=list_of_groups, vector_dataframe=TFIDF_VEC)
    else:
        logging.info("Building bag of mutations vectorizer for all sequences...")
        if args.reference_name:
            logging.info(f"The sequence named as {args.reference_name} will be used as reference sequence.")
            reference_sequence = ALN[args.reference_name]
            BOM_VEC = create_bag_of_mutations(fasta_dict=ALN, reference_sequence=reference_sequence)
        else:
            logging.info("No reference sequence was defined. Therefore, the consensus sequence will be considered as the reference.")
            BOM_VEC = create_bag_of_mutations(fasta_dict=ALN, reference_sequence=None)
            logging.info("Building bag of mutations representations for each group...")
        list_of_groups_tfidf = create_TFIDF_groups(list_of_groups=list_of_groups, vector_dataframe=BOM_VEC)
    
    # --- Execução do Subsampling ---
    FINAL = pd.DataFrame()
    logging.info("Running subsampling across all groups...")
    
    for i in tqdm(range(len(list_of_groups)), desc="Subsampling groups"):
        group_date, group_clade, group_df, group_alloc = list_of_groups[i]
        
        if group_alloc == 0 or len(group_df) == 0:
            continue
            
        alloc = allocate_proportional(group_df, list_of_groups_tfidf[i], target_N=int(group_alloc),
                                            region_col='region',
                                            alpha=args.alpha,
                                            preserve_max_share=args.preserve_max_share,
                                            use_diversity=args.use_diversity,
                                            beta=args.beta)

        meta_by_region, tfidf_by_region = split_tfidf_by_region(group_df, list_of_groups_tfidf[i], region_col='region')
        
        for region, region_meta_df in meta_by_region.items():
            region_alloc = alloc.get(region, 0)
            if region_alloc == 0:
                continue
            
            if args.use_GA:
                region_tfidf_df = tfidf_by_region[region]
                indexes = run_filter_sequences(alloc_values=int(region_alloc), tfidf_dataframe=region_tfidf_df,
                                            min_pop=args.min_pop, max_pop=args.max_pop, 
                                            precompute_metric=args.precompute_metric,
                                            reduce_dim_before_distance=args.reduce_dim_before_distance,
                                            generations=args.generations,
                                            tournament_size=args.tournament_size,
                                            elitism=args.elitism,
                                            mutation_rate=args.mutation_rate,
                                            random_state=args.random_seed,
                                            verbose=False,
                                            crossover_rate=args.crossover_rate)
            else:
                indexes = run_hybrid_sampling(alloc_values=int(region_alloc), metadata_subgroup=region_meta_df, random_state=args.random_seed)
            
            tmp = region_meta_df.loc[indexes]
            FINAL = pd.concat([FINAL, tmp], ignore_index=True)
    
    logging.info("Subsampling has finished.")    
    logging.info(f"Total sequences in final dataset: {len(FINAL)}")
    
    # --- Salvando o Resultado ---
    logging.info(f"Saving selected sequences to {args.output}...")
    with open(args.output, "w") as output_file:
        for _, row in FINAL.iterrows():
            name = row["name"]
            sequence = str(row["sequence"]).replace("-","")

            if args.keep_header:
            
                output_file.write(f">{name}\n{sequence}\n")    
            
            else:

                clade = row["clade"]
                region = row["region"]
                date = row["date"].strftime("%Y-%m-%d")
                output_file.write(f">{name}|{clade}|{region}|{date}\n{sequence}\n")
    
    logging.info("Calculating the clade frequence by dates for subsampled Dataset...")    
    
    subsampling_freq_by_month = streamPlot(df=FINAL, clade_color_dict= clade_color_dict, output="subsampling_clade_frequencies_by_date.pdf" , figsize=(24, 8))
    
    logging.info(f"{subsampling_freq_by_month.mean(axis=1)}")

    logging.info("Calculating mean absolute and mean squared errors...")    
    mae, mse = error_calculate(df1= original_freq_by_month, df2= subsampling_freq_by_month)
    
    logging.info("Saving mean absolute error...")
    
    for k, v in tqdm(mae.items(), desc="Saving mean absolute error"):
        save_dict2json(Dict=v, filename=f"{k}.json")

    logging.info(f"mae overall {mae['mae_overall']}")

    logging.info("Saving mean squared error...")

    for k, v in tqdm(mse.items(), desc="Saving mean squared error"):
        
        save_dict2json(Dict=v, filename=f"{k}.json")
    
    logging.info(f"mse overall {mse['mse_overall']}")

    logging.info("Saving final FINAL_DATASET.tsv...")
    
    FINAL.to_csv("FINAL_DATASET.tsv", sep="\t", index=False)

    logging.info("GenoSieve run completed successfully.")

if __name__ == "__main__":
    main()