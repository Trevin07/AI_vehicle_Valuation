from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Inside Docker, this is where your scripts live
scripts_dir = "/opt/airflow/scripts"

default_args = {
    'owner': 'cdb_intern',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'cdb_vehicle_pipeline_v5',
    default_args=default_args,
    description='Full Pipeline: Join, Clean, IQR, NLP Mapping',
    schedule_interval=None,
    catchup=False,
    tags=['CDB', 'Vehicle_Valuation'],
) as dag:

    # 1. Run the main pipeline orchestrator or individual files
    # Here we run your 4 main files in the exact order you requested
    
    join_task = BashOperator(
        task_id='1_Join_Files',
        bash_command=f'python {scripts_dir}/joined.py'
    )

    clean_task = BashOperator(
        task_id='2_Data_Cleaning_IQR',
        bash_command=f'python {scripts_dir}/data_pre.py'
    )

    violation_task = BashOperator(
        task_id='3_Mean_Median_Filling',
        bash_command=f'python {scripts_dir}/final_violation_check.py'
    )

    mapping_task = BashOperator(
        task_id='4_NLP_ERP_Mapping',
        bash_command=f'python {scripts_dir}/mapping_final.py'
    )

    # Define the sequence
    join_task >> clean_task >> violation_task >> mapping_task