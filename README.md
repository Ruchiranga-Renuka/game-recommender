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


## 📝 Notes

- Make sure `games.csv` is present in the project folder before running.
- To push to GitHub, you must have Git configured and be authenticated (SSH key or personal access token).


<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/d95cc757-f1ba-4803-b7f4-6c17d334a732" />
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/fa571d5d-012a-4570-8bdc-2ad9978042b1" />
