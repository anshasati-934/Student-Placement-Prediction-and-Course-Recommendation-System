# 🎓 Edu-vision: Student Placement Prediction and Recommendation System

A machine learning-based web application that predicts the placement probability of students and recommends skills, roles, and learning paths tailored to specific companies and job profiles. Designed to assist students in improving their employability through personalized data-driven insights.

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Screenshots](#screenshots)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## 📖 About the Project

This project is aimed at enhancing student career readiness through data-driven predictions and skill recommendations. By analyzing academic scores, participation, projects, and other parameters, the system determines placement probabilities and suggests a skill roadmap to align with industry expectations.

It also uses real-time job data and clustering mechanisms to offer interactive dashboards and analytics to students and placement cells.

## ✨ Features

- 🎯 Placement probability prediction using ML models
- 🔍 Skill and company-based recommendation engine
- 📊 Real-time student clustering using Mini-Batch KMeans
- 📈 Visualization using dynamic Dash-based dashboards
- 🔐 Role-based access control (Admin, Student, Recruiter)
- 🔔 Personalized notifications and career suggestions
- 🧠 Cosine Similarity for resume-job matching
- ⚡ Real-time data updates (graphs, clustering after every 50 entries)
- 📬 Feedback integration loop for continuous improvement

## 🛠️ Tech Stack

| Component            | Technology                     |
|---------------------|---------------------------------|
| Backend             | Python, Flask                   |
| Frontend            | HTML, CSS, Bootstrap, JS        |
| Machine Learning    | Scikit-learn, Pandas, NumPy     |
| Visualization       | Plotly, Dash                    |
| Database            | SQLite / MySQL                  |
| APIs                | Adzuna API (Jobs), OpenAI (NLP) |
| Hosting             | Render / PythonAnywhere         |

## 📸 Screenshots

| Page                            | Description                          |
|---------------------------------|--------------------------------------|
| 🧾 Registration Page            | User registration & login            |
| 📈 Dashboard                    | Student dashboard with stats         |
| 🧠 Prediction & Recommendation  | Prediction result with skill roadmap |
| 🔍 Company-based Skill Suggest  | Based on job market API              |
| 📊 Real-Time Clustering         | Dash-powered visual clusters         |

> Note: Images are available in the `/screenshots` folder of this repo.

## 🏗️ Project Architecture

1. User logs in or registers (role-based access).
2. Student data is preprocessed & clustered.
3. ML model predicts placement probability.
4. Cosine similarity used to match resumes with job descriptions.
5. Real-time job data fetched via Adzuna API.
6. Recommendations and visualizations rendered dynamically.
7. Dash dashboards update every 50 new entries.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/edu-vision-placement-predictor.git
   cd edu-vision-placement-predictor
Install the dependencies:

bash
Always show details

Copy
pip install -r requirements.txt
Run the Flask App:

bash
Always show details

Copy
python app.py
Access at:

cpp
Always show details

Copy
http://127.0.0.1:5000
🧪 Model Evaluation
Clustering Technique: MiniBatchKMeans

Silhouette Score: 0.42 (avg)

Feature Engineering: Academic scores, Projects, Participation

Similarity Analysis: Cosine Similarity

Evaluation Metrics: Accuracy, Precision, Recall

👨‍💻 Contributors
Anurag Chorghade (28)

Ansh Asati (26)

Akshat Shah (23)

Abishek Choudhary (21)

Supervised by:

Dr. Shubhangi Neware
Shri Ramdeobaba College of Engineering and Management, Nagpur

🙏 Acknowledgements
We are thankful to the Department of Computer Science and Engineering at RCOEM for guidance and resources. Special thanks to our guide Dr. Shubhangi Neware for her mentorship throughout this project.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
