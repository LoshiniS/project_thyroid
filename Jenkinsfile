pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run Streamlit App') {
            steps {
                bat 'streamlit run app.py'
            }
        }
    }
}
