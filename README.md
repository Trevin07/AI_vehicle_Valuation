# AI-Driven Vehicle Valuation & ERP Mapping Pipeline 🚘 ➡️ 🏎️ 

An automated end-to-end data engineering pipeline built to process vehicle market data, implement statistical outlier filtering, and map internal database models to a standardized ERP dataset using Natural Language Processing (NLP).

## Overview
This project solves the challenge of inconsistent vehicle naming conventions. It automates the transition from raw, messy data into a standardized format ready for valuation modeling. The system is containerized using **Docker** and orchestrated by **Apache Airflow** for full visibility and monitoring.

##  Tech Stack
* **Orchestration:** Apache Airflow (DAG-based workflow)
* **Containerization:** Docker & Docker Compose (WSL2 Backend)
* **Data Processing:** Python (Pandas, OpenPyXL)
* **Statistical Filtering:** IQR (Interquartile Range) for price outlier detection
* **Matching Engine:** Custom-built Fuzzy Matcher with Token Variant Generation (replaces standard NLP for higher speed/predictability).
* **Algorithms:** Heuristic-based Scoring (Coverage, Precision, Order Bonus, and Substring matching).
* **Environment:** Linux-based Docker containers running on Windows

##  Project Structure
* `dags/`: Airflow DAG definitions (`vehicle_dag.py`)
* `scripts/`: Core Python modules for each pipeline stage
* `docker-compose.yaml`: Infrastructure configuration
* `flow.txt`: Technical roadmap of the data transformations

## Pipeline Stages
1.  **Data Joining:** Merges disparate vehicle data sources into a standardized master file.
2.  **Preprocessing & IQR Filtering:** Cleans text data and applies statistical bounds to remove unrealistic price entries (Outliers).
3.  **Violation Checks & Imputation:** Identifies data gaps and applies Mean/Median filling to ensure dataset completeness.
4.  **Model Mapping:** Uses a custom `ModelIndex` with variant generation (e.g., mapping 'A4' to '4A' or '318i' to '318') to match DB entries against a standardized ERP Excel (.xlsx) dataset.

##  Getting Started
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Trevin07/AI_vehicle_Valuation.git](https://github.com/Trevin07/AI_vehicle_Valuation.git)
    ```
2.  **Environment Setup:** Ensure your `scripts/` folder contains the necessary `.csv` and `.xlsx` (ERP) files.
3.  **Launch via Docker:**
    ```bash
    docker compose up airflow-init
    docker compose up -d
    ```
4.  **Monitor:** Access the UI at `http://localhost:8080` to trigger and track the pipeline.
