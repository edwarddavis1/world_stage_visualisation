# %%
from scipy import sparse
import scipy.linalg as LA
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
countries_to_remove = ["Donetsk People's Republic", "Hezbollah", "Lebanon ( Hezbollah)",
                       "Lugansk People's Republic", "Venezuela (Guaido government)", "Western Sahara"]
country_corrections_dict = {"Bahamas": "The Bahamas", "St. Lucia": "Saint Lucia",
                            "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
                            "The Gambia": "Gambia", }

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
        alliance_list = []
        for country_with_spaces in line[line.find('\t') + 2: -1].split(deli):
            country = country_with_spaces.strip()
            if country in country_corrections_dict:
                country = country_corrections_dict[country]
            if country not in countries_to_remove:
                alliance_list.append(country)

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
country_df["name"] = country_df["name"].replace(
    {"Saint Kitts and Nevis": "St. Kitts and Nevis"})

# gdp
# source: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
# Source for taiwan: https://www.statista.com/statistics/727589/gross-domestic-product-gdp-in-taiwan/#:~:text=In%202020%2C%20Taiwan's%20gross%20domestic,around%20668.16%20billion%20U.S.%20dollars.
# Source for palestine: world bank
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

# UN General Assembly vote to condemn the Russian attack on Ukraine
# source: wikipedia
un_vote_df_raw = pd.read_csv(
    "un_vote_russia_attack_on_ukraine.txt", delimiter="\t")
list_of_states = []
vote_list = []
for states in un_vote_df_raw["States"].values:
    list_of_states_for_vote = states.split(", ")
    list_of_states.extend(list_of_states_for_vote)
    vote_list.extend([un_vote_df_raw[un_vote_df_raw["States"] == states]
                      ["Vote"].values[0]] * len(list_of_states_for_vote))

un_vote_df = pd.DataFrame({"Country": list_of_states, "Vote": vote_list})

# Region of each country
# source: https://statisticstimes.com/geography/countries-by-continents.php
region_df = pd.read_csv("country_with_region.txt", delimiter="\t")
region_df = region_df[["Country or Area", "Region 1", "Continent"]]
region_df.columns = ["Country", "Region", "Continent"]

# Religion
# source: https://worldpopulationreview.com/country-rankings/religion-by-country
# Note this is as of 2010
religion_df_raw = pd.read_csv("country_by_religion.csv")
religions = religion_df_raw.columns[1:]
most_popular_religion = []
population_most_popular_religion = []
for country in religion_df_raw["country"]:
    most_popular_religion_idx = np.argmax(
        religion_df_raw[religion_df_raw["country"] == country].values[0][1:])
    most_popular_religion.append(religions[most_popular_religion_idx])
    population_most_popular_religion.append(
        religion_df_raw[religion_df_raw["country"] == country][religions[most_popular_religion_idx]].values[0])

religion_df = religion_df_raw.copy()
religion_df["Most Popular Religion"] = most_popular_religion
religion_df["Population with Most Popular Religion"] = population_most_popular_religion

# %% [markdown]
# Convert to adjacency matrirx
# %%
# Get list of all countries
countries_with_dups = [
    item for subitem in alliance_list_list for item in subitem]
countries = list(set(countries_with_dups))
countries.sort()

# Remove unwanted countries
countries = list(set(countries) - set(countries_to_remove))

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
gdps = []
for country in countries:
    try:
        gdps.append(gdp_df[gdp_df["Country"]
                           == country]["GDP"].values[0])
    except:
        gdps.append(0)
        print(country)

# %%
pops = []
for country in countries:
    try:
        pops.append(country_df[country_df["name"] ==
                               country]["pop2022"].values[0])
    except:
        # if country == "Taiwan":
        #     pops.append("Not in UN")
        # elif country == "Palestine":
        #     pops.append("Not in UN")
        # else:
        print(country)
        # pops.append(None)
# %%

most_popular_religion = []
population_popular_religion = []
for country in countries:
    try:
        most_popular_religion.append(religion_df[religion_df["country"] ==
                                                 country]["Most Popular Religion"].values[0])
        # Population figures are in thousands of people
        population_popular_religion.append(religion_df[religion_df["country"] ==
                                                       country]["Population with Most Popular Religion"].values[0] / 1000)
    except:
        print(country)
        most_popular_religion.append("No Data")
        population_popular_religion.append(0)

# percentage_population_with_religion = np.array(
#     population_popular_religion) / np.array(pops)

# %%
un_votes = []
for country in countries:
    try:
        un_votes.append(un_vote_df[un_vote_df["Country"] ==
                                   country]["Vote"].values[0])
    except:
        if country == "Taiwan":
            un_votes.append("Not in UN")
        elif country == "Palestine":
            un_votes.append("Not in UN")
        else:
            print(country)
            un_votes.append(None)


regions = []
continents = []
for country in countries:
    try:
        regions.append(region_df[region_df["Country"] ==
                                 country]["Region"].values[0])
        continents.append(region_df[region_df["Country"] ==
                                    country]["Continent"].values[0])

    except:
        if country == "Taiwan":
            regions.append("Eastern Asia")
            continents.append("Asia")
        elif country == "Palestine":
            regions.append("Western Asia")
            continents.append("Asia")
        else:
            print(country)
            regions.append(None)
            continents.append(None)

# %% [markdown]
# Compute spectral embedding
# %%
# u, s, vt = np.linalg.svd(A)
# d = 10
# xa = u[:, 0:d] @ np.diag(np.sqrt(s[0:d]))
# xa_umap = umap.UMAP(n_components=2).fit_transform(xa)

# p = 2
# q = 0.5
p = 1
q = 1
n2v_obj = nodevectors.Node2Vec(
    n_components=25,
    return_weight=1/p,
    neighbor_weight=1/q,
    epochs=5000,
    walklen=100,
    w2vparams={"window": 10, "negative": 5, "iter": 10,
               "batch_words": 128}
)
xa = n2v_obj.fit_transform(A @ A.T)
xa_pca = PCA(n_components=8).fit_transform(xa)
# xa_umap = umap.UMAP(n_components=2).fit_transform(xa_pca)

# ggvec_obj = nodevectors.GGVec(
#     n_components=25,
#     max_epoch=1000
# )
# xa = ggvec_obj.fit_transform(A)
# xa_pca = PCA(n_components=8).fit_transform(xa)


# def safe_inv_sqrt(a, tol=1e-12):
#     """Computes the inverse square root, but returns zero if the result is either infinity
#     or below a tolerance"""
#     with np.errstate(divide="ignore"):
#         b = 1 / np.sqrt(a)
#     b[np.isinf(b)] = 0
#     b[a < tol] = 0
#     return b


# def to_laplacian(A, regulariser=0):
#     """Constructs the (regularised) symmetric Laplacian.
#     """
#     left_degrees = np.reshape(np.asarray(A.sum(axis=1)), (-1,))
#     right_degrees = np.reshape(np.asarray(A.sum(axis=0)), (-1,))
#     if regulariser == 'auto':
#         regulariser = np.mean(np.concatenate((left_degrees, right_degrees)))
#     left_degrees_inv_sqrt = safe_inv_sqrt(left_degrees + regulariser)
#     right_degrees_inv_sqrt = safe_inv_sqrt(right_degrees + regulariser)
#     L = sparse.diags(
#         left_degrees_inv_sqrt) @ A @ sparse.diags(right_degrees_inv_sqrt)
#     return L


# # Construct (regularised) Laplacian matrix
# L = to_laplacian(A, regulariser=100)
# # L = A

# # Compute spectral embedding
# d = 8
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
# # xadf = pd.DataFrame(xa_pca)
xadf = pd.read_csv("final_xadf.csv")
# xadf.columns = ["Dimension {}".format(i+1) for i in range(xadf.shape[1])]
# xadf["Country"] = countries
# xadf["NATO Membership"] = np.where(
#     np.isin(countries, nato_df["Country"]), "Yes", "No")

# # Use estimated gdp for north korea (estimated by the bank of south korea)
# xadf["GDP"] = np.nan_to_num(gdps, nan=28500.)
# xadf["Is GDP Estimate"] = np.where(
#     xadf["Country"] == "North Korea", True, False)
# xadf["UN Vote on Ukraine"] = un_votes
# xadf["Continent"] = continents

# # Number of alliances
# num_alliances = np.zeros((n,))
# for alliance in alliance_list_list:
#     for country in xadf["Country"]:
#         if country in alliance:
#             num_alliances[country_id[country]] += 1

# xadf["Number of Alliances"] = num_alliances

# xadf["Dominant Religion"] = most_popular_religion

# # # Combined GDP of all allies (essentially who has the most powerful allies)
# # comb_gdp_of_allies = np.zeros((n, ))
# # for country in xadf["Country"].values:
# #     list_of_allies = np.where(A[country_id[country], :] > 0)[0]
# #     if gdps[country_id[country]] != None and gdps[country_id[country]] > 0:
# #         comb_gdp_of_allies[country_id[country]] += gdps[country_id[country]]

# # # Also add their own gdp
# # for country in xadf["Country"].values:
# #     comb_gdp_of_allies[country_id[country]] += gdps[country_id[country]]

# # xadf["Combined GDP of Allies"] = comb_gdp_of_allies

# xadf["GDP"] = xadf["GDP"].astype(float)
# xadf["GDP"] = np.sqrt(xadf["GDP"])

# # Number of allied countries
# num_allied_countries = []
# for country in xadf["Country"].values:
#     num_allied_countries.append(sum(A[country_id[country], :]))
# xadf["Number of Allies"] = num_allied_countries

# xadf = xadf.sort_values(by=["GDP", "Country"], ascending=False)


# xadf.to_csv("xa_50_epoch_df.csv")

fig = px.scatter(xadf, x="Dimension 1", y="Dimension 2",
                 #   color="NATO Membership",
                 #    color="UN Vote on Ukraine",
                 color="Continent",
                 #  color="Dominant Religion",
                 # color="Country",
                 #  text="Country",
                 #  #  size="Combined GDP of Allies",
                 # #  size="Number of Allies",
                 size="GDP",
                 #    symbol="Is GDP Estimate",
                 template="plotly_white",
                 hover_data=["Country", "GDP",
                             "NATO Membership", "Number of Allies", "Number of Alliances",
                             "UN Vote on Ukraine"],
                 )
fig.update_traces(marker=dict(
    # symbol=list(np.where(xadf["Is GDP Estimate"] == False, "circle", "x-thin")),
    line=dict(width=1, color="darkslateblue"),
    sizemode="area",
    sizeref=2. * max(xadf["GDP"])/(80. ** 2),
    #   sizeref=2. * \
    #   max(xadf["Combined GDP of Allies"])/(160. ** 2),
    #   sizemin=4,
),
    textposition="top center",
)
fig.update_layout(
    xaxis=dict(
        title="",
        linecolor="white",  # Sets color of X-axis line
        # showgrid=False,  # Removes X-axis grid lines
        showticklabels=False,
    ),
    yaxis=dict(
        title="",
        linecolor="white",  # Sets color of Y-axis line
        # showgrid=False,  # Removes Y-axis grid lines
        showticklabels=False,
    ),
)
# fig.show()
fig.show(renderer="browser")
# %% [markdown]
# Poking
# %%
south_america_df = xadf[(xadf["Dimension 1"] > 7) & (xadf["Dimension 2"] > 13)]
plotly_default_colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
                          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
south_america_df = south_america_df.sort_values(by=["GDP"], ascending=False)
bar = px.bar(south_america_df, x="Country", y="GDP",
             #  color="Continent",
             #  color_discrete_sequence=[
             #  plotly_default_colours[0], plotly_default_colours[3]],
             )
bar.show()
# %%
europe_df = xadf[(xadf["Dimension 1"] > 9) & (xadf["Dimension 2"] < -9)]
plotly_default_colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
                          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
europe_df = europe_df.sort_values(by=["GDP"], ascending=False)
bar = px.bar(europe_df, x="Country", y="GDP",
             #  color="Continent",
             #  color_discrete_sequence=[
             #  plotly_default_colours[0], plotly_default_colours[3]],
             )
bar.show()
# %%
pacific_df = xadf[(xadf["Dimension 2"] < 7) & (xadf["Dimension 2"] > 4)]
plotly_default_colours = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
                          '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
pacific_df = pacific_df.sort_values(by=["GDP"], ascending=False)
bar = px.bar(pacific_df, x="Country", y="GDP",
             #  color="Continent",
             #  color_discrete_sequence=[
             #  plotly_default_colours[0], plotly_default_colours[3]],
             )
bar.show()
# %%
