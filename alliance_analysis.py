# %%
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import nodevectors
import umap

pio.renderers.default = "notebook_connected"

"""
Countries like "The Bahamas" and "Bahamas" are treated separately - fix this!
"""
# %% [markdowningdom]
# Read in data
# %%
# Alliance data
# source: https://en.m.wikipedia.org/wiki/List_of_military_alliances
# note: I have removed new zealand from the ANZUS alliance as it is "paritally suspended"
# I have replaced each occurance of "UK" and "USA" to "United Kingdom" and "United States" - as the dataset seems to use both.
# Similar with "UAE"
# Changed "Czechia" to "Czech Republic"
# "Cabo Verde" to "Cape Verde"
textfile = open("raw_wiki_alliance_data.txt", "r")
lines = textfile.readlines()
deli = "•"
alliance_name_list = []
alliance_list_list = []
for line in lines:
    if deli in line:
        alliance_name_list.append(line[0:line.find('\t')])
        alliance_list = [country.strip()
                         for country in line[line.find('\t') + 2: -1].split(deli)]
        alliance_list_list.append(alliance_list)

alliance_df = pd.DataFrame()
alliance_df["Alliance"] = alliance_name_list
alliance_df["Countries"] = alliance_list_list

# Country data
# source: https://worldpopulationreview.com/countries
# "DR Congo" to "Democratic Republic of the Congo"
country_df = pd.read_csv("country_sizes.csv")
country_df["name"] = country_df["name"].replace(
    {"DR Congo": "Democratic Republic of the Congo"})
country_df["name"] = country_df["name"].replace(
    {"Bahamas": "The Bahamas"})
country_df["name"] = country_df["name"].replace(
    {"St. Lucia": "Saint Lucia"})
country_df["name"] = country_df["name"].replace(
    {"St. Vincent and the Grenadines": "Saint Vincent and the Grenadines"})
country_df["name"] = country_df["name"].replace(
    {"The Gambia": "Gambia"})

# gdp
# source: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
# Source for taiwan: https://www.statista.com/statistics/727589/gross-domestic-product-gdp-in-taiwan/#:~:text=In%202020%2C%20Taiwan's%20gross%20domestic,around%20668.16%20billion%20U.S.%20dollars.
# Note that the most recent year of gdp is not 2022 - but we are including alliances from 2022
# gdp_df = pd.read_csv("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4019306.csv", skiprows=[0,1,2,3])
gdp_df = pd.read_csv("country_most_recent_gdp.txt",
                     delimiter="\t", header=None)
gdp_df.columns = ["Country", "Most Recent Year", "GDP", "None"]
gdp_df = gdp_df[["Country", "Most Recent Year", "GDP"]]

# NATO
# source: https://en.m.wikipedia.org/wiki/Member_states_of_NATO
nato_df = pd.read_csv("nato.txt", header=None, names=["Country"])

# # Check country names
# for country in countries:
#     if country not in coords_df["name"].values:
#         print(country)

# Country coordinates
# source: https://developers.google.com/public-data/docs/canonical/countries_csv
coords_df = pd.read_csv("country_coordinates.txt", delimiter="\t")

# %% [markdown]
# Convert to adjacency matrix
# %%
# Get list of all countries
countries_with_dups = [
    item for subitem in alliance_list_list for item in subitem]
countries = list(set(countries_with_dups))
countries.sort()
country_id = {country: i for i, country in enumerate(countries)}
id_country = {i: country for i, country in enumerate(countries)}
n = len(countries)

A = np.zeros((n, n))
for prog, connected_countries in enumerate(alliance_list_list):
    for country in connected_countries:
        connected_countries_copy = connected_countries.copy()
        connected_countries_copy.remove(country)
        for other_country in connected_countries_copy:
            A[country_id[country], country_id[other_country]] += 1
# %%
# THESE NEED TO BE DEALT WITH!
# populations = []
# for country in countries:
#     try:
#         populations.append(country_df[country_df["name"]
#                                       == country]["pop2022"].values[0])
#     except:
#         populations.append(0)
#         print(country)

gdps = []
for country in countries:
    try:
        gdps.append(gdp_df[gdp_df["Country"]
                           == country]["GDP"].values[0])
    except:
        gdps.append(0)
        print(country)
# %%
# THESE NEED TO BE DEALT WITH!
lats = []
for country in countries:
    try:
        lats.append(coords_df[coords_df["name"] ==
                              country]["latitude"].values[0])
    except:
        print(country)
        lats.append(None)

lons = []
for country in countries:
    try:
        lons.append(coords_df[coords_df["name"] ==
                              country]["longitude"].values[0])
    except:
        print(country)
        lons.append(None)

# %% [markdown]
# Compute spectral embedding
# %%
# u, s, vt = np.linalg.svd(A)
# d = 10
# xa = u[:, 0:d] @ np.diag(np.sqrt(s[0:d]))
# xa_umap = umap.UMAP(n_components=2).fit_transform(xa)

n2v_obj = nodevectors.Node2Vec(n_components=25)
xa = n2v_obj.fit_transform(A)
xa_pca = PCA(n_components=8).fit_transform(xa)
xa_umap = umap.UMAP(n_components=2).fit_transform(xa_pca)

# # Construct (regularised) Laplacian matrix
# L = to_laplacian(A, regulariser=10000)

# # Compute spectral embedding
# L_vals, L_vecs = sparse.linalg.eigs(L, d*2 + 2)
# idx = np.abs(L_vals).argsort()[::-1]
# L_vals = L_vals[idx]
# L_vecs = np.real(L_vecs[:, idx])

# # Remove lamda = 1 eigenvalues (as Laplacian matrices always have these, dilation means there are two)
# U = np.real(L_vecs[:, 0::2][:, 1:])
# S = np.diag(abs(L_vals[0::2][1:]))
# embedding = U @ LA.sqrtm(S)

# # Divide by sqrt of node degree
# degree_corrected_ya_vecs = []
# degree = np.reshape(np.asarray(A.sum(axis=0)), (-1,))
# for i in range(degree.shape[0]):
#     if degree[i] > 10e-8:
#         degree_corrected_ya_vecs.append(
#             np.divide(embedding[i, :], np.sqrt(degree[i])))
#     else:
#         degree_corrected_ya_vecs.append(
#             np.zeros(embedding[i, :].shape))

# xa = np.vstack(degree_corrected_ya_vecs)
# xa_umap = umap.UMAP(n_components=2).fit_transform(xa)
# %%
xadf = pd.DataFrame(xa_pca)
xadf.columns = ["Dimension {}".format(i+1) for i in range(xadf.shape[1])]
xadf["Country"] = countries
xadf["NATO Membership"] = np.where(
    np.isin(countries, nato_df["Country"]), "Yes", "No")
xadf["Latitude"] = lats
xadf["Longitude"] = lons
xadf["GDP"] = gdps
xadf = xadf[xadf["GDP"] != None]
xadf["GDP"] = xadf["GDP"].astype(float)
xadf = xadf[xadf["GDP"] > 0]
# xadf = xadf[xadf["GDP"] > np.mean(xadf["GDP"].values)]
# xadf["GDP"] = np.sqrt(xadf["GDP"])

fig = px.scatter(xadf, x="Dimension 1", y="Dimension 2",
                 #   color="Country",
                 #   color="NATO Membership",
                 size="GDP",
                 )
# fig.update_traces(marker=dict(size=12,
#                               line=dict(width=2,
#                                         color=(np.where(xadf["NATO Membership"].values == "Yes", 'DarkSlateGrey', 'White'))),
#                   selector=dict(mode='markers'))
fig.update_traces(marker=dict(line=dict(width=1.5, color=np.where(
    xadf["NATO Membership"].values == "Yes", "darkslateblue", "white"))))
# fig.update_layout(
#     xaxis=dict(
#         title="",
#         linecolor="white",  # Sets color of X-axis line
#         showgrid=False  # Removes X-axis grid lines
#     ),
#     yaxis=dict(
#         title="",
#         linecolor="white",  # Sets color of Y-axis line
#         showgrid=False,  # Removes Y-axis grid lines
#     ),
#
# )
fig.show()
# %% [markdown]
# Poking
# %%
n2v_obj = nodevectors.Node2Vec(n_components=3)
xa = n2v_obj.fit_transform(A)

xadf = pd.DataFrame(xa)
xadf.columns = ["Dimension {}".format(i+1) for i in range(xadf.shape[1])]
xadf["Country"] = countries
fig = px.scatter_3d(xadf, x="Dimension 1", y="Dimension 2", z="Dimension 3",
                    color="Country",
                    )
fig.show()
# %%
