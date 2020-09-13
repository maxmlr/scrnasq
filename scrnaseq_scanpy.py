#!/usr/bin/env python3

# %%
## NOTE Initialization
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# python
import sys
import os
import csv
import shutil
import distutils
import itertools
import configparser
from pathlib import Path

# required packages: python-igraph, louvain, scanpy
import igraph
import louvain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc

# %%
## NOTE configuration
config = configparser.ConfigParser()
config.read('config.ini')

# set random seed
random_seed = config.getint('general', 'random_seed')
np.random.seed(random_seed)

# get paths from config file
raw_base = Path(config.get('filesystem', 'raw_folder'))
user_base = Path(config.get('filesystem', 'user_folder'))
out_base = Path(config.get('filesystem', 'out_folder'))
sample_name = config.get('sample', 'sample_name')
sample_src = config.get('sample', 'sample_src')

# set required paths
sample_data = raw_base / sample_name / sample_src
cellphonedb_data_folder = user_base / 'cpdb'
signature_genes_folder = user_base / 'signatures'
signature_CAFs_folder = user_base / 'CAF_signatures'
cpdb_cluster_annotations_folder = user_base / 'cpdb_cluster_annotations'
user_clusters_folder = user_base / 'user_defined_cluster'
user_embedding_folder = user_base / 'embedding'

# set out folders
out_folder = out_base / sample_name
cellphonedb_out = out_folder / 'cpdb'
data_folder = out_folder / 'data'
deg_analysis = out_folder / 'deg'

# create required output folders
cellphonedb_out.mkdir(parents=True, exist_ok=True)
data_folder.mkdir(parents=True, exist_ok=True)
deg_analysis.mkdir(parents=True, exist_ok=True)

# visualization
sns.set_style('white')
sns.mpl.pyplot.rcParams['savefig.facecolor'] = 'w'
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=300)
sc.settings.figdir = out_folder

# %%
## NOTE Define some helper functions
def load_gene_signature(name, base=signature_genes_folder, suffix='txt'):
    with (base / f'{name}.{suffix}').open() as fin_signature:
        return [l.strip() for l in fin_signature.readlines() if not l.startswith('#')]

def load_cpdb_runs(sample_id, name):
    with (cpdb_cluster_annotations_folder / sample_id / f'{name}.txt').open() as fin_run:
        return [l.strip().split(',') for l in fin_run.readlines() if not l.startswith('#')]

def print_highlite(content, border=None):
    hline = ''.join(['-' for _ in content]) if border is None else '-' *  border
    print(f'\n+{hline}--+\n{"  " if border is None else ""}{content.strip()}\n+{hline}--+\n')

def query_yes_no(question, default='no'):
    if default is None:
        prompt = " [y/n] "
    elif default == 'yes':
        prompt = " [Y/n] "
    elif default == 'no':
        prompt = " [y/N] "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == '':
                return default == 'yes'
            else:
                return distutils.util.strtobool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

# %%
## NOTE read sample data
print_highlite(f'Scanpy v{sc.__version__}')
print_highlite(f'Analysing sample: {sample_name}\n')

adata = sc.read_10x_h5(sample_data)
adata.raw = adata
# ------------ WARNING ------------ #
# raw is overwritten later,
# after filtering & transformation
# --------------------------------- #
adata.var_names_make_unique()
print_highlite(f'\nRaw data: {adata.n_obs} cells, {adata.n_vars} genes')

# %%
## NOTE generate QC data and add total counts per cell as observations-annotation to adata
adata.var['mt'] = adata.var_names.str.startswith('mt-')
adata.var['rb'] = adata.var_names.str.startswith(('Rps', 'Rpl'))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'rb'], inplace=True)

# %%
## NOTE filter by mitochondrial
if config.getboolean('filters', 'filter_mitochondrial_cells'):
    filter_mito_upper_pct = config.getfloat('filters', 'filter_mito_upper_pct')
    adata = adata[adata.obs.pct_counts_mt < filter_mito_upper_pct, :]
    print_highlite(f'After removing cells with >= {filter_mito_upper_pct}% mitochondrial content: {adata.n_obs} cells, {adata.n_vars} genes')

# %%
## NOTE filter by ribosomal
if config.getboolean('filters', 'filter_ribosomal_cells'):
    filter_ribo_upper_pct = config.getfloat('filters', 'filter_ribo_upper_pct')
    adata = adata[adata.obs.pct_counts_rb < filter_ribo_upper_pct, :]
    print_highlite(f'After removing cells with >= {filter_ribo_upper_pct}% ribosomal content: {adata.n_obs} cells, {adata.n_vars} genes')

# %%
## NOTE filter by min cells / genes
filter_min_genes_cnt = config.getint('filters', 'filter_min_genes_cnt')
filter_min_cells_cnt = config.getint('filters', 'filter_min_cells_cnt')
sc.pp.filter_cells(adata, min_genes=filter_min_genes_cnt)
sc.pp.filter_genes(adata, min_cells=filter_min_cells_cnt)
print_highlite(f'After removing cells with < {filter_min_genes_cnt} genes and cells with < {filter_min_cells_cnt} cells: {adata.n_obs} cells, {adata.n_vars} genes')

# %%
## NOTE remove all mitochondrial genes
if config.getboolean('filters', 'remove_mitochondrial_genes'):
    mito_genes_cnt = sum(adata.var['mt'])
    non_mito_genes_list = adata.var_names[~adata.var['mt']]
    adata = adata[:, non_mito_genes_list]
    print_highlite(f'Removed {mito_genes_cnt} mitochondrial genes')

# %%
## NOTE remove all ribosomal genes
if config.getboolean('filters', 'remove_ribosomal_genes'):
    ribo_genes_cnt = sum(adata.var['rb'])
    non_ribo_genes_list = adata.var_names[~adata.var['rb']]
    adata = adata[:, non_ribo_genes_list]
    print_highlite(f'Removed {ribo_genes_cnt} ribosomal genes')

# %%
## NOTE Plot QC-features
sc.settings.figdir = out_folder / 'quality_control'
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save=f'.genes_vs_counts.pdf', show=False)

# %%
## NOTE transform data: normalize, log-transform
adata_raw = adata.copy()
sc.pp.normalize_total(adata, target_sum=config.getfloat('preprocessing', 'transformation_norm_scale_factor'))
adata_normalized = adata.copy()
sc.pp.log1p(adata)
adata.raw = adata

# %%
## NOTE compute variable genes
sc.pp.highly_variable_genes(
    adata,
    min_mean=config.getfloat('analysis.variable', 'find_variable_genes_min_mean'),
    max_mean=config.getfloat('analysis.variable', 'find_variable_genes_max_mean'),
    min_disp=config.getfloat('analysis.variable', 'find_variable_genes_min_disp'),
    max_disp=config.getfloat('analysis.variable', 'find_variable_genes_max_disp')
)
sc.pl.highly_variable_genes(adata, save=f'.highly_variable_genes.pdf', show=False)
plt.close()
adata = adata[:, adata.var.highly_variable]
print_highlite(f'Number of highly variable genes found: {len(adata.var.gene_ids)}')

# %%
## NOTE transform data: regress, scale
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata)

# %%
## NOTE run and plot PCA
embeddings_pca_npcs = config.getint('preprocessing', 'embeddings_pca_npcs')
sc.tl.pca(adata, n_comps=embeddings_pca_npcs, random_state=random_seed)
sc.pl.pca(adata, components = ['1,2'], save=f'.pca.pdf', show=False)

# %%
## NOTE check for most relevant PCs: Heatmaps, Elbow plot
if config.getboolean('preprocessing', 'analyze_relevant_pcas'):
    for pc in range(1, embeddings_pca_npcs + 1):
        g = adata.varm['PCs'][:,pc-1]
        o = np.argsort(g)
        sel = np.concatenate((o[:10],o[-10:])).tolist()
        emb = adata.obsm['X_pca'][:,pc-1]
        # order by position on that pc
        tempdata = adata[np.argsort(emb),]
        sc.pl.heatmap(tempdata, var_names = adata.var.gene_ids[sel].index.tolist(), swap_axes = True, use_raw=False, save=f'.heatmap.pdf', show=False)
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=embeddings_pca_npcs, save=f'.pca_variance_ratio.pdf', show=False)

# %%
## NOTE construct neighborhood graph
sc.pp.neighbors(adata, n_pcs=config.getint('preprocessing', 'selected_pcs'), n_neighbors=config.getint('preprocessing', 'graph_neighbors'), metric="euclidean", random_state=random_seed) 

# %%
## NOTE Clustering: louvain
louvain_resolution = config.getfloat('preprocessing', 'louvain_resolution')
sc.tl.louvain(adata, resolution=louvain_resolution, key_added=f"louvain_{louvain_resolution}", random_state=random_seed)

# %%
## NOTE generate UMAP embedding or if load an existing one from csv file
umap_embedding = user_embedding_folder / f'{sample_name}_umap.csv'
if umap_embedding.exists():
    print_highlite('Loading existing UMAP embedding...')
    X_umap = np.array(pd.read_csv(umap_embedding)[['x', 'y']], dtype=float)
    if X_umap.shape[0] == adata.shape[0]:
        adata.obsm['X_umap'] = X_umap
    else:
        print_highlite(f'WARNING: Loaded umap embedding has different number of input features: {X_umap.shape[0]} vs. {adata.shape[0]} - skipping')
if adata.obsm.get('X_umap') is None:
    sc.tl.umap(adata, min_dist=config.getfloat('preprocessing', 'umap_min_dist'), spread=config.getfloat('preprocessing', 'umap_spread'), random_state=random_seed)
umap_data = pd.DataFrame(adata.obsm['X_umap'], columns = ['x', 'y'])
umap_data['cluster'] = adata.obs[f'louvain_{louvain_resolution}'].values
umap_data['labels'] = adata.obs[f'louvain_{louvain_resolution}'].index
umap_data.to_csv(data_folder / 'umap.csv', index=False)

# %%
## NOTE manually select cluster of interest
sc.settings.figdir = out_folder / 'clusters_and_gene_signatures'
umap_main = sc.pl.umap(adata, color=[f'louvain_{louvain_resolution}'], title="UMAP", color_map="magma", use_raw=False, save=f'.louvain_clustering.pdf', show=False)
sc.settings.figdir = out_folder
print('\n')
user_clusters_set = query_yes_no(f'Do you want to manually select a cluster in the UMAP embedding (see {sc.settings.figdir / "umap.louvain_clustering.pdf"})?') if config.getboolean('preprocessing', 'new_user_cluster_option') else False
if user_clusters_set:
    plt.close()
    sc.settings.set_figure_params(dpi=75)
    from plot_select import main as plot_select
    plot_select(str(data_folder / 'umap.csv'))
    plt.close()
    sc.settings.set_figure_params(dpi=300)
    user_input = input("Enter selection label: ")
    user_cluster_name = user_input if user_input != '' else "user_selection"
    shutil.move(data_folder / 'umap_selected.csv', user_clusters_folder / f'{user_cluster_name}.csv')

# %%
## NOTE load user selected cells and annotate as seperate cluster
if config.getboolean('preprocessing', 'load_user_clusters') or user_clusters_set:
    user_clusters = [f.stem for f in user_clusters_folder.glob("*.csv")]
    for user_cluster_name in user_clusters:
        user_cluster = user_clusters_folder / f'{user_cluster_name}.csv'
        user_selected_cells = None
        adata_user_selected = None
        if user_cluster.exists() and user_cluster_name not in adata.obs[f'louvain_{louvain_resolution}'].cat.categories:
            user_selected_cells = pd.read_csv(user_cluster)
            adata_user_selected = adata[adata.obs.index.isin(user_selected_cells.selected)]
            adata.obs[f'louvain_{louvain_resolution}'].cat.add_categories(user_cluster_name, inplace=True)
            adata.obs.loc[adata.obs.index.isin(user_selected_cells.selected) , f'louvain_{louvain_resolution}'] = user_cluster_name

# %%
## NOTE plot UMAP embedding
sc.settings.figdir = out_folder / 'clusters_and_gene_signatures'
umap_main = sc.pl.umap(adata, color=[f'louvain_{louvain_resolution}'], title="UMAP", color_map="magma", use_raw=False, save=f'.louvain_clustering.pdf', show=False)
sc.settings.figdir = out_folder

# %%
## NOTE plot signatures
sc.settings.figdir = out_folder / 'clusters_and_gene_signatures'
all_raw_genes = set(adata.raw.var.index)
all_variable_genes = set(adata.var.index)
user_signatures = [f.stem for f in signature_genes_folder.glob("*.txt")]
for sig_name in user_signatures:
    loaded_signature = load_gene_signature(sig_name)
    missing_in_raw = set(loaded_signature) - all_raw_genes
    overlap = set(loaded_signature) & all_raw_genes
    if len(missing_in_raw):
        print_highlite(f'WARNING [user_signatures]: MISSING GENES IN RAW DATA -> {missing_in_raw} for signature: {sig_name}')
    print_highlite(f'Plotting signature {sig_name}: {overlap} ({len(overlap)} genes)')
    sc.pl.umap(adata, color=overlap, use_raw=True, color_map="viridis", vmin=0, save=f'.{sig_name}.signature.pdf', show=False)
    sc.tl.score_genes(adata, gene_list=loaded_signature, score_name=f'{sig_name}_ref', use_raw=False, random_state=random_seed)
    sc.tl.score_genes(adata, gene_list=loaded_signature, score_name=f'{sig_name}_ref_raw', use_raw=True, random_state=random_seed)
    #fig.suptitle(f'{sig_name} signature', fontsize=16)
sc.settings.figdir = out_folder

# %%
## NOTE load pre-refined *-CAF signatures
CAFs_signatures = [f.stem for f in signature_CAFs_folder.glob("*.csv")]
CAFs_signatures_dict = {}
for sig_name in CAFs_signatures:
    CAFs_signatures_dict[sig_name] = load_gene_signature(sig_name, base=signature_CAFs_folder, suffix='csv')
signature = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in CAFs_signatures_dict.items()]))

# %%
## NOTE check overlap of *-CAF signatures
for sig_name in signature.columns:
    sig = signature[sig_name].dropna()
    info = (
        f'{sig_name}: #total genes        : {len(set(sig))}\n'
        f'{sig_name}: missing in all (raw): {len(set(sig) - all_raw_genes)}/{len(set(sig))}\n'
        f'{sig_name}: missing in variable : {len(set(sig) - all_variable_genes)}/{len(set(sig))}\n'
        f'{sig_name}: overlap in variable : {len(set(sig) & all_variable_genes)}/{len(set(sig))}\n'
    )
    print_highlite(info, 50)

# %%
## NOTE compute *-CAF signatures score
for sig_name in signature.columns:
    sig = signature[sig_name].dropna()
    sig = sig[sig.isin(set(sig) & all_variable_genes)]
    sc.tl.score_genes(adata, gene_list=sig, score_name=f'{sig_name}_ref', use_raw=False, random_state=random_seed)
    sc.tl.score_genes(adata, gene_list=sig, score_name=f'{sig_name}_ref_raw', use_raw=True, random_state=random_seed)

# %%
## NOTE generate *-CAF signatures scores violin plots
sc.settings.figdir = out_folder / 'CAF_signatures'
sc.pl.violin(
    adata, keys=[f'{s}_ref' for s in signature.columns], save=f'.markers.pancaf.all_caf.refined.pdf', rotation=90, show=False
)
sc.settings.figdir = out_folder

# %%
## NOTE generate *-CAF signatures scores umap plots
sc.settings.figdir = out_folder / 'CAF_signatures'
sc.pl.umap(
    adata, color=[f'{s}_ref' for s in signature.columns], color_map='viridis', vmin=0, save=f'.markers.pancaf.all_caf.refined.pdf', show=False
)
sc.settings.figdir = out_folder

# %%
## NOTE generate *-CAF signatures signature genes umap plots
if config.getboolean('signatures', 'generate_CAF_per_gene_umap_plots'):
    print_highlite('Generating *-CAF per gene umap plots...')
    sc.settings.figdir = out_folder / 'CAF_signatures'
    for sig_name in signature.columns:
        sig = signature[sig_name].dropna()
        sc.pl.umap(
            adata, color=sig.values, color_map='viridis', vmin=0, save=f'.markers.pancaf.{sig_name}.refined.pdf', show=False
        )
    sc.settings.figdir = out_folder

# %%
## NOTE generate cellphonedb input files
if config.getboolean('cellphonedb', 'generate_input_data'):
    HMD_HumanPhenotype_colnames = ["HumanSymbol", "HumanEntrez", "HomoloGeneID", "Presence", "MouseSymbol", "MGIMarkerAccessionID", "HighlevelMammalianPhenotypeID", "NA"]
    hmd = pd.read_csv(cellphonedb_data_folder / "HMD_HumanPhenotype.rpt.txt", delimiter='\t', names=HMD_HumanPhenotype_colnames)
    hmd = hmd.drop_duplicates(subset=['HumanSymbol']).drop_duplicates(subset=['MouseSymbol'])

    user_cpdb_runs = (cpdb_cluster_annotations_folder / sample_name ).glob("*.csv")
    for cpdb_run in user_cpdb_runs:
        print_highlite(f'Generating cellphonedb input files for {cpdb_run.stem}...')
        user_annotations = {}
        with cpdb_run.open() as fin:
            annot_reader = csv.reader(fin)
            header = next(annot_reader)
            for row in annot_reader:
                user_annotations[row[header.index('cluster_id')]] = row[header.index('cluster_annotation')]
        for cluster_id, cluster_annot in user_annotations.items():
            adata.obs.loc[adata.obs[f'louvain_{louvain_resolution}'] == cluster_id, f'louvain_{louvain_resolution}_annot'] = cluster_annot
        adata_cpdb_reduced = adata[adata.obs[f'louvain_{louvain_resolution}'].isin(user_annotations.keys())]
        adata_cpdb_reduced.obs[f'louvain_{louvain_resolution}_annot'].to_csv(cellphonedb_out / f'{sample_name}_{cpdb_run.stem}_meta.tsv', index_label='Cell', header=['cell_type'], sep='\t')

        adata_normalized_tmp = adata_normalized[adata.obs[f'louvain_{louvain_resolution}'].isin(user_annotations.keys())]
        count_matrix = adata_normalized_tmp.copy().T.to_df()
        count_matrix.index.name = 'Gene'

        count_matrix = pd.merge(count_matrix, hmd[['MouseSymbol', 'HumanSymbol']], left_on='Gene', right_on='MouseSymbol', how='inner')
        count_matrix.index = count_matrix.HumanSymbol
        count_matrix.index.name = 'Gene'
        count_matrix.drop(['HumanSymbol', 'MouseSymbol'], axis=1, inplace=True)

        count_matrix.to_csv(cellphonedb_out / f'{sample_name}_{cpdb_run.stem}_count.tsv', sep='\t')

        # docker commands to run celphonedb
        cellphonedb_analysis = f'docker run --rm -v "{cellphonedb_data_folder}":/mnt/db -v "{cellphonedb_out}":/mnt/cpdb cellphonedb method statistical_analysis /mnt/cpdb/{sample_name}_{cpdb_run.stem}_meta.tsv /mnt/cpdb/{sample_name}_{cpdb_run.stem}_count.tsv --counts-data gene_name --database /mnt/db/cellphonedb_user_all_2020-05-23-14_17.db --threshold 0.01 --output-path /mnt/cpdb/out/{cpdb_run.stem} --iterations 1'
        cellphonedb_dotplot = f'docker run --rm -v "{cellphonedb_data_folder}":/mnt/db -v "{cellphonedb_out}":/mnt/cpdb cellphonedb plot dot_plot --output-path /mnt/cpdb/out/{cpdb_run.stem} --means-path /mnt/cpdb/out/{cpdb_run.stem}/means.txt --pvalues-path /mnt/cpdb/out/{cpdb_run.stem}/pvalues.txt  # optional: --rows /mnt/cpdb/out/{cpdb_run.stem}/dot_plot_rows.txt --columns /mnt/cpdb/out/{cpdb_run.stem}/dot_plot_cols.txt'
        cellphonedb_heatmap = f'docker run --rm -v "{cellphonedb_data_folder}":/mnt/db -v "{cellphonedb_out}":/mnt/cpdb cellphonedb plot heatmap_plot --output-path /mnt/cpdb/out/{cpdb_run.stem} --pvalues-path /mnt/cpdb/out/{cpdb_run.stem}/pvalues.txt /mnt/cpdb/{sample_name}_{cpdb_run.stem}_meta.tsv'
        print_highlite(f'cellphonedb commands:\n\n - analysis:\n{cellphonedb_analysis}\n\n - dot plot:\n{cellphonedb_dotplot}\n\n - heatmap:\n{cellphonedb_heatmap}', 100)

# %%
## NOTE DEG: compute
sc.settings.figdir = out_folder / 'deg'
adata_deg = adata.copy()
raw_tmp = adata_raw.raw.to_adata()
raw_tmp.var_names_make_unique()
adata_deg.raw = raw_tmp

clusters_of_interest = tuple(config.get('analysis.deg', 'clusters_of_interest').split(','))

deg_top_genes_plot = config.getint('analysis.deg', 'deg_top_genes_plot')
test_method = config.get('analysis.deg', 'test_method')

clusters_of_interest_subset = adata_deg[adata_deg.obs[f'louvain_{louvain_resolution}'].isin(clusters_of_interest)]
raw_mito_ribo_genes = clusters_of_interest_subset.raw.var_names.str.startswith(('mt-', 'Rps', 'Rpl'))
clusters_of_interest_subset.raw = clusters_of_interest_subset.raw[:, ~raw_mito_ribo_genes].to_adata()

sc.tl.rank_genes_groups(clusters_of_interest_subset, groupby=f'louvain_{louvain_resolution}', groups=clusters_of_interest, use_raw=True, method=test_method, n_genes=clusters_of_interest_subset.raw.shape[1])
result_deg_raw = clusters_of_interest_subset.uns['rank_genes_groups']
groups_deg_raw = result_deg_raw['names'].dtype.names
deg_raw_df = pd.DataFrame({
    group: result_deg_raw[key][group] for group in groups_deg_raw for key in ['names']
})

sc.pl.rank_genes_groups(clusters_of_interest_subset, save=f'.deg_names.pdf', show=False)
sc.pl.rank_genes_groups_dotplot(clusters_of_interest_subset, n_genes=deg_top_genes_plot, save=f'.deg_dotplots_counts.pdf', show=False)
sc.pl.rank_genes_groups_dotplot(clusters_of_interest_subset, n_genes=deg_top_genes_plot, values_to_plot='logfoldchanges', min_logfoldchange=3, vmax=7, vmin=-7, cmap='bwr', groups=None, save=f'.deg_dotplots_foldchange.pdf', show=False)

sc.tl.dendrogram(clusters_of_interest_subset, groupby=f'louvain_{louvain_resolution}')
sc.pl.rank_genes_groups_tracksplot(clusters_of_interest_subset, groups=clusters_of_interest, save=f'.deg_Tracksplot_counts.pdf', show=False)

# %%
## NOTE DEG: get all
# TODO Only keep genes with foldchange delta above threshold ?

min_fold_change = 1
min_in_group_fraction = 0
max_out_group_fraction = 1.01
sc.tl.filter_rank_genes_groups(clusters_of_interest_subset, min_in_group_fraction=min_in_group_fraction, min_fold_change=min_fold_change, max_out_group_fraction=max_out_group_fraction)
result = clusters_of_interest_subset.uns['rank_genes_groups_filtered']
groups = result['names'].dtype.names

deg_dict_nas = {group: result[key][group] for group in groups for key in ['names']}
deg_dict = {group: pd.Series(pd.Series(genes).dropna().values) for group, genes in deg_dict_nas.items()}
deg_df = pd.DataFrame(deg_dict)
for col in deg_df.columns:
    print_highlite(f'{col}: {len(deg_df[col].dropna())}')

deg_df.index.name = 'Genes'
deg_df.columns.name = 'Groups'
deg_df.to_csv(deg_analysis / f'DEG_all_{test_method}.csv')

# %%
## NOTE DEG: get mutually exclusive genes

min_fold_change = 1
min_in_group_fraction = 0.8
max_out_group_fraction = 0.5
sc.tl.filter_rank_genes_groups(clusters_of_interest_subset, key_added='rank_genes_groups_filtered_mutually_exclusive', min_in_group_fraction=min_in_group_fraction, min_fold_change=min_fold_change, max_out_group_fraction=max_out_group_fraction)

result = clusters_of_interest_subset.uns['rank_genes_groups_filtered_mutually_exclusive']
groups = result['names'].dtype.names

deg_dict_nas = {group: result[key][group] for group in groups for key in ['names']}
deg_dict = {group: pd.Series(pd.Series(genes).dropna().values) for group, genes in deg_dict_nas.items()}
deg_me_raw_df = pd.DataFrame(deg_dict)

deg_me_dict = {}
for col in deg_me_raw_df.columns:
    current = pd.Series(deg_me_raw_df[col].dropna().values)
    remaining = pd.Series(deg_me_raw_df.drop(col, axis=1).values.flatten()).dropna()
    overlap = current.index.intersection(remaining.index)
    deg_me_dict[col] = pd.Series(current.drop(overlap).values)
deg_me_df = pd.DataFrame(deg_me_dict)
for col in deg_me_df.columns:
    me_genes = deg_me_df[col].dropna()
    print_highlite(f'{col}: {len(me_genes)}')

deg_me_df.index.name = 'Genes'
deg_me_df.columns.name = 'Groups'
deg_me_df.to_csv(deg_analysis / f'DEG_mutually_exclusive_{test_method}.csv')

# %%
## NOTE DEG: plot genes states per cluster
clusters_of_interest = config.get('analysis.deg.states', 'clusters_of_interest').split(',')
genes_of_interest = config.get('analysis.deg.states', 'genes_of_interest').split(',')

# for gene in genes_of_interest:
#     adata_clusters = adata[adata.obs[f'louvain_{louvain_resolution}'].isin(clusters_of_interest)]
#     adata_clusters_gene = adata_clusters.raw.to_adata()[:, [gene]]
#     # threshold = np.mean(adata_clusters_gene.X[:,adata_clusters_gene.var.index == gene])
#     threshold = np.percentile(adata_clusters_gene.X[:,adata_clusters_gene.var.index == gene].toarray().flatten(), 75)
#     adata_clusters_gene.obs[f'{gene}_state'] = None
#     adata_clusters_gene.obs.loc[adata_clusters_gene.X.toarray().flatten() >= threshold, f'{gene}_state'] = 'high'
#     adata_clusters_gene.obs.loc[adata_clusters_gene.X.toarray().flatten() < threshold, f'{gene}_state'] = 'low'
#     sc.pl.umap(adata_clusters_gene, color=[f'{gene}_state'], title=gene, color_map="viridis", use_raw=False)
#     sc.pl.umap(adata_clusters_gene, color=[f'{gene}'], title=gene, color_map="viridis", use_raw=False)

for gene_1, gene_2 in itertools.combinations(genes_of_interest, 2):
    adata_clusters = adata[adata.obs[f'louvain_{louvain_resolution}'].isin(clusters_of_interest)]

    adata_clusters_gene_1 = adata_clusters.raw.to_adata()[:, [gene_1]]
    # threshold = np.mean(adata_clusters_gene_1.X[:,adata_clusters_gene_1.var.index == gene_1])
    threshold = np.percentile(adata_clusters_gene_1.X[:,adata_clusters_gene_1.var.index == gene_1].toarray().flatten(), 75)
    adata_clusters_gene_1.obs[f'{gene_1}_state'] = None
    adata_clusters_gene_1.obs.loc[adata_clusters_gene_1.X.toarray().flatten() >= threshold, f'{gene_1}_state'] = 'high'
    adata_clusters_gene_1.obs.loc[adata_clusters_gene_1.X.toarray().flatten() < threshold, f'{gene_1}_state'] = 'low'

    adata_clusters_gene_2 = adata_clusters.raw.to_adata()[:, [gene_2]]
    # threshold = np.mean(adata_clusters_gene_2.X[:,adata_clusters_gene_2.var.index == gene_2])
    threshold = np.percentile(adata_clusters_gene_2.X[:,adata_clusters_gene_2.var.index == gene_2].toarray().flatten(), 75)
    adata_clusters_gene_2.obs[f'{gene_2}_state'] = None
    adata_clusters_gene_2.obs.loc[adata_clusters_gene_2.X.toarray().flatten() >= threshold, f'{gene_2}_state'] = 'high'
    adata_clusters_gene_2.obs.loc[adata_clusters_gene_2.X.toarray().flatten() < threshold, f'{gene_2}_state'] = 'low'

    # different_state = adata_clusters_gene_1.obs[f'{gene_1}_state'] != adata_clusters_gene_2.obs[f'{gene_2}_state']
    # adata_clusters_gene_1.obs[f'{gene_1}_state_differs'] = different_state.map({True: 'high/low', False: 'low/low & high/high'})

    adata_clusters_gene_1.obs[f'{gene_1}_state_differs'] = None
    adata_clusters_gene_1.obs.loc[list(((adata_clusters_gene_1.obs[f'{gene_1}_state'] == 'high') & (adata_clusters_gene_2.obs[f'{gene_2}_state'] == 'low')).values), f'{gene_1}_state_differs'] = 'high/low'
    adata_clusters_gene_1.obs.loc[list(((adata_clusters_gene_1.obs[f'{gene_1}_state'] == 'low') & (adata_clusters_gene_2.obs[f'{gene_2}_state'] == 'high')).values), f'{gene_1}_state_differs'] = 'low/high'
    adata_clusters_gene_1.obs.loc[list(((adata_clusters_gene_1.obs[f'{gene_1}_state'] == 'high') & (adata_clusters_gene_2.obs[f'{gene_2}_state'] == 'high')).values), f'{gene_1}_state_differs'] = 'high/high'
    adata_clusters_gene_1.obs.loc[list(((adata_clusters_gene_1.obs[f'{gene_1}_state'] == 'low') & (adata_clusters_gene_2.obs[f'{gene_2}_state'] == 'low')).values), f'{gene_1}_state_differs'] = 'low/low'
    # adata_clusters_gene_1.obs.loc[adata_clusters_gene_1.obs[f'{gene_1}_state'] == adata_clusters_gene_2.obs[f'{gene_2}_state'], f'{gene_1}_state_differs'] = 'same'
    
    sc.pl.umap(adata_clusters_gene_1, color=[f'{gene_1}_state_differs'], title=f'{gene_1}/{gene_2}' , palette=['blue', 'lightgrey', 'lightgrey', 'lightgrey'], save=f'.states.{gene_1}_vs_{gene_2}.pdf', show=False)
