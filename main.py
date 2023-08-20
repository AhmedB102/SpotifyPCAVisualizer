import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
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


def modified_pca_plotly(fig, songs_list, row, col):
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

    for i, row_data in df.iterrows():
        song_name = row_data["name"]
        album = row_data["album"]
        song_url = row_data["url"]
        hover_text = f'<a href="{song_url}">{song_name}</a>'


        fig.add_trace(
            go.Scatter(
                x=[principalDf["principal component 1"].iloc[i]],
                y=[principalDf["principal component 2"].iloc[i]],
                mode="markers+text",
                marker=dict(size=10),
                text=hover_text,
                hovertemplate=album,
                hovertext=hover_text,
                textfont_size=9,
                textposition="top center",
                customdata=[[album, song_url]],
                marker_symbol="circle-open",
            ),
            row=row,
            col=col
        )

    # Update the layout for this subplot
    fig.update_xaxes(title_text="Principal Component 1", row=row, col=col)
    fig.update_yaxes(title_text="Principal Component 2", row=row, col=col)



def modified_tsne_plotly(fig, songs_list, row, col):
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

    # Apply t-SNE and reduce the data to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    tsne_components = tsne.fit_transform(X_scaled)

    # Convert to DataFrame for convenience
    tsneDf = pd.DataFrame(
        data=tsne_components,
        columns=["t-SNE dimension 1", "t-SNE dimension 2"],
    )

    for i, row_data in df.iterrows():
        song_name = row_data["name"]
        song_url = row_data["url"]
        hover_text = f'<a href="{song_url}">{song_name}</a>'

        fig.add_trace(
            go.Scatter(
                x=[tsneDf["t-SNE dimension 1"].iloc[i]],
                y=[tsneDf["t-SNE dimension 2"].iloc[i]],
                mode="markers+text",
                marker=dict(size=10),
                text=hover_text,
                # hovertemplate='%{text}',
                hovertext=hover_text,
                textfont_size=9,
                textposition="top center",
            ),
            row=row,
            col=col
        )

    # Update the layout for this subplot
    fig.update_xaxes(title_text="t-SNE Dimension 1", row=row, col=col)
    fig.update_yaxes(title_text="t-SNE Dimension 2", row=row, col=col)

def combined_plot(songs_list):
    # Create a subplot layout
    fig = make_subplots(rows=1, cols=2, subplot_titles=('PCA', 't-SNE'))

    # Add PCA plot to the first subplot
    modified_pca_plotly(fig, songs_list, row=1, col=1)
    
    # Add t-SNE plot to the second subplot
    modified_tsne_plotly(fig, songs_list, row=1, col=2)

    # Update layout for better appearance, and add title
    fig.update_layout(title="PCA and t-SNE Visualization of Audio Features")
    
    # Show the combined figure
    fig.show()

def search_and_plot(search_query):
    songs_data = fetch_songs_and_features(search_query)
    combined_plot(songs_data)


if __name__ == "__main__":
    # The script will prompt for an artist's name and then fetch and plot the songs
    artist_name = input("Enter the artist's name to search and visualize songs: ")
    search_and_plot(artist_name)
