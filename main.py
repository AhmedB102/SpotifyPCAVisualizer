import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config


def fetch_songs_and_features(search_query):
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=config.CLIENT_ID,
            client_secret=config.CLIENT_SECRET,
            username=config.USERNAME,
            redirect_uri=config.REDIRECT_URI,
            scope="user-top-read",
        )
    )
    results = sp.search(search_query, limit=50, type="track")["tracks"]
    print(f"Search for {search_query} succesfully completed")
    ids = [item["id"] for item in results["items"]]
    songs_list = []

    # Fetch audio features for all songs in one go
    print("Fetching audio details ....")
    features = sp.audio_features(ids)

    for idx, song_id in enumerate(ids):
        print(f"Fetching details for song {idx}")
        # get song's meta data
        meta = sp.track(song_id)

        # Create song info dictionary
        song_info = {
            "id": song_id,
            "album": meta["album"]["name"],
            "name": meta["name"],
            "artist": ", ".join(
                [singer_name["name"] for singer_name in meta["artists"]]
            ),
            "url": meta["external_urls"]["spotify"],
            "explicit": meta["explicit"],
            "popularity": meta["popularity"],
            # Adding audio features
            "danceability": features[idx]["danceability"],
            "energy": features[idx]["energy"],
            "loudness": features[idx]["loudness"],
            "speechiness": features[idx]["speechiness"],
            "acousticness": features[idx]["acousticness"],
            "instrumentalness": features[idx]["instrumentalness"],
            "liveness": features[idx]["liveness"],
            "valence": features[idx]["valence"],
            "tempo": features[idx]["tempo"],
            "duration_ms": features[idx]["duration_ms"]
            / 60000,  # Convert ms to minutes
        }

        # Append song info to the list
        songs_list.append(song_info)
    return songs_list


def plot_songs_with_pca_plotly(songs_list):
    # Extract audio features
    features = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]

    # Create a DataFrame from the songs list for easier manipulation
    df = pd.DataFrame(songs_list)
    X = df[features]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA and reduce the data to 2 dimensions
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_scaled)

    # Convert to DataFrame for convenience
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )

    # Create a plotly figure
    fig = go.Figure()

    for i, row in df.iterrows():
        song_name = row["name"]
        album = row["album"]
        song_url = row["url"]
        hover_text = f"{song_name}<br>{song_url}"

        fig.add_trace(
            go.Scatter(
                x=[principalDf["principal component 1"].iloc[i]],
                y=[principalDf["principal component 2"].iloc[i]],
                mode="markers+text",
                marker=dict(size=10),
                text=song_name,
                hovertext=hover_text,
                hoverinfo="text",
                textfont_size=9,
                textposition="top center",
                customdata=[[album, song_url]],
                marker_symbol="circle-open",
            )
        )

    # Extract the explained variance by each principal component
    explained_variance = pca.explained_variance_ratio_

    # Add annotations with the explained variance
    annotations = [
        dict(
            xref="paper",
            yref="paper",
            x=0.8,
            y=1.05,
            xanchor="left",
            yanchor="bottom",
            text=f"Explained Variance by Principal Component 1: {explained_variance[0]:.2%}",
            font=dict(family="Arial", size=12, color="black"),
            showarrow=False,
        ),
        dict(
            xref="paper",
            yref="paper",
            x=0.8,
            y=1.0,
            xanchor="left",
            yanchor="bottom",
            text=f"Explained Variance by Principal Component 2: {explained_variance[1]:.2%}",
            font=dict(family="Arial", size=12, color="black"),
            showarrow=False,
        ),
    ]

    # Update layout for better appearance, and add title
    fig.update_layout(
        title="2 Component PCA of Audio Features",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        showlegend=False,
        annotations=annotations,
    )

    fig.show()


def search_and_plot_pca(search_query):
    songs_data = fetch_songs_and_features(search_query)
    plot_songs_with_pca_plotly(songs_data)


if __name__ == "__main__":
    # The script will prompt for an artist's name and then fetch and plot the songs
    artist_name = input("Enter the artist's name to search and visualize songs: ")
    search_and_plot_pca(artist_name)
