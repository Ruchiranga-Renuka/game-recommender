# 🎮 Game Recommender

A simple Streamlit-based Steam game recommender that suggests games based on your preferred genre and platform, powered by pandas and scikit-learn.

## 🚀 Run Locally

**1. Install dependencies**
```powershell
python -m pip install -r requirements.txt
```

**2. Start the app**
```powershell
python -m streamlit run app.py
```

## ⚙️ How It Works

1. Select a **genre** and **platform**
2. The app filters games from the dataset using **pandas**
3. **Scikit-learn** computes similarity scores
4. A list of recommended Steam games is displayed

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) — web app interface
- [Pandas](https://pandas.pydata.org/) — data processing
- [Scikit-learn](https://scikit-learn.org/) — recommendation logic

## 📁 Project Structure
game-recommender/
├── app.py                  # Main Streamlit app
├── game_recommender.ipynb  # Data exploration notebook
├── games.csv               # Steam game dataset
├── requirements.txt        # Dependencies
└── test_app.py             # Unit tests

## 📝 Notes

- Make sure `games.csv` is present in the project folder before running.
- To push to GitHub, you must have Git configured and be authenticated (SSH key or personal access token).
