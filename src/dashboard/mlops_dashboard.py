#!/usr/bin/env python3
"""
MLOps Movie Recommendation Dashboard

Web-based dashboard for managing the entire MLOps pipeline including:
- Model training and monitoring
- API management
- Data pipeline control
- Performance analytics
- System health monitoring

Author: MLOps Movie Recommendation System
"""

import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="MLOps Movie Recommendation Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


class MLOpsDashboard:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.project_root = Path(__file__).parent.parent.parent
        self.monitoring_dir = Path("monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)

    def check_api_status(self):
        """Check API health status"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return False, {"error": str(e)}

    def get_model_stats(self):
        """Get model statistics"""
        try:
            response = requests.get(f"{self.api_base_url}/model/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception:
            return None

    def test_recommendation(self, movie_id: int = 278):
        """Test recommendation functionality"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/recommendations",
                json={"movie_id": movie_id, "top_k": 5},
                timeout=10,
            )
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return True, {
                    "response_time_ms": response_time,
                    "recommendations": data.get("recommendations", []),
                    "movie_id": movie_id,
                }
            else:
                return False, {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return False, {"error": str(e)}

    def test_search(self, query: str = "action"):
        """Test search functionality"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/search",
                json={"query": query, "top_k": 3},
                timeout=10,
            )
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return True, {
                    "response_time_ms": response_time,
                    "results": data.get("results", []),
                    "query": query,
                }
            else:
                return False, {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return False, {"error": str(e)}

    def get_system_metrics(self):
        """Get system resource metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "load_average": os.getloadavg()[0] if hasattr(os, "getloadavg") else 0,
        }

    def run_data_pipeline(self):
        """Run data processing pipeline"""
        try:
            result = subprocess.run(
                ["python", "-m", "src.data.preprocess_data"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Process timeout"
        except Exception as e:
            return False, "", str(e)

    def run_model_training(self):
        """Run model training"""
        try:
            result = subprocess.run(
                ["python", "-m", "src.models.train_model"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Process timeout"
        except Exception as e:
            return False, "", str(e)


def main():
    dashboard = MLOpsDashboard()

    # Header
    st.markdown(
        '<h1 class="main-header">🎬 MLOps Movie Recommendation Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "🏠 Overview",
            "📊 Model Monitoring",
            "🔧 Pipeline Management",
            "🧪 Testing",
            "📈 Analytics",
            "⚙️ System",
        ],
    )

    if page == "🏠 Overview":
        show_overview(dashboard)
    elif page == "📊 Model Monitoring":
        show_model_monitoring(dashboard)
    elif page == "🔧 Pipeline Management":
        show_pipeline_management(dashboard)
    elif page == "🧪 Testing":
        show_testing(dashboard)
    elif page == "📈 Analytics":
        show_analytics(dashboard)
    elif page == "⚙️ System":
        show_system_metrics(dashboard)


def show_overview(dashboard):
    st.header("System Overview")

    # API Status
    col1, col2, col3, col4 = st.columns(4)

    api_healthy, api_data = dashboard.check_api_status()

    with col1:
        if api_healthy:
            st.metric("API Status", "🟢 Healthy", delta="Online")
        else:
            st.metric("API Status", "🔴 Down", delta="Offline")

    # Model Stats
    model_stats = dashboard.get_model_stats()
    if model_stats:
        with col2:
            st.metric("Total Movies", f"{model_stats['total_movies']:,}")
        with col3:
            st.metric("Vocabulary Size", f"{model_stats['vocabulary_size']:,}")
        with col4:
            st.metric("Cache Size", model_stats["cache_size"])

    # Recent Activity
    st.subheader("Recent Activity")

    if api_healthy:
        st.success("✅ API is running and responding normally")
        if api_data.get("uptime_seconds"):
            uptime_hours = api_data["uptime_seconds"] / 3600
            st.info(f"🕐 API uptime: {uptime_hours:.1f} hours")
    else:
        st.error("❌ API is not responding")
        st.error(f"Error: {api_data.get('error', 'Unknown error')}")

    # Quick Actions
    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Status"):
            st.rerun()

    with col2:
        if st.button("🧪 Run Quick Test"):
            with st.spinner("Testing API..."):
                success, result = dashboard.test_recommendation()
                if success:
                    st.success(f"✅ Test passed ({result['response_time_ms']:.1f}ms)")
                else:
                    st.error(f"❌ Test failed: {result['error']}")

    with col3:
        if st.button("📊 View Metrics"):
            st.switch_page("📊 Model Monitoring")


def show_model_monitoring(dashboard):
    st.header("Model Monitoring")

    # Real-time metrics
    st.subheader("Real-time Performance")

    # Create columns for metrics
    col1, col2, col3 = st.columns(3)

    # Test recommendations
    with st.spinner("Testing recommendations..."):
        rec_success, rec_result = dashboard.test_recommendation()

    with col1:
        if rec_success:
            st.metric(
                "Recommendation Response",
                f"{rec_result['response_time_ms']:.1f}ms",
                delta="Fast",
            )
            st.success(
                f"✅ Generated {len(rec_result['recommendations'])} recommendations"
            )
        else:
            st.metric("Recommendation Response", "Error", delta="Failed")
            st.error(f"❌ {rec_result['error']}")

    # Test search
    with st.spinner("Testing search..."):
        search_success, search_result = dashboard.test_search()

    with col2:
        if search_success:
            st.metric(
                "Search Response",
                f"{search_result['response_time_ms']:.1f}ms",
                delta="Fast",
            )
            st.success(f"✅ Found {len(search_result['results'])} results")
        else:
            st.metric("Search Response", "Error", delta="Failed")
            st.error(f"❌ {search_result['error']}")

    # Model stats
    model_stats = dashboard.get_model_stats()
    with col3:
        if model_stats:
            st.metric("Model Status", "🟢 Loaded", delta="Ready")
            st.info(
                f"Matrix: {model_stats['matrix_shape'][0]}×{model_stats['matrix_shape'][1]}"
            )
        else:
            st.metric("Model Status", "🔴 Not Loaded", delta="Error")

    # Performance Charts
    st.subheader("Performance Trends")

    # Generate sample data for demo
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7), end=datetime.now(), freq="H"
    )
    response_times = np.random.normal(50, 10, len(dates))
    success_rates = np.random.uniform(0.95, 1.0, len(dates))

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "response_time": response_times,
            "success_rate": success_rates,
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(df, x="timestamp", y="response_time", title="Response Time Trend")
        fig.update_layout(yaxis_title="Response Time (ms)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(df, x="timestamp", y="success_rate", title="Success Rate Trend")
        fig.update_layout(yaxis_title="Success Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Model Quality Metrics
    st.subheader("Model Quality")

    if rec_success and rec_result["recommendations"]:
        recs = rec_result["recommendations"]
        similarities = [r["similarity"] for r in recs]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Avg Similarity", f"{np.mean(similarities):.3f}")
        with col2:
            st.metric("Min Similarity", f"{np.min(similarities):.3f}")
        with col3:
            st.metric("Max Similarity", f"{np.max(similarities):.3f}")

        # Similarity distribution
        fig = px.histogram(x=similarities, title="Similarity Score Distribution")
        fig.update_layout(xaxis_title="Similarity Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)


def show_pipeline_management(dashboard):
    st.header("Pipeline Management")

    # Data Pipeline
    st.subheader("Data Pipeline")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Data Processing Pipeline**")
        if st.button("🔄 Run Data Processing"):
            with st.spinner("Running data processing pipeline..."):
                success, stdout, stderr = dashboard.run_data_pipeline()

                if success:
                    st.success("✅ Data processing completed successfully!")
                    with st.expander("View Output"):
                        st.code(stdout)
                else:
                    st.error("❌ Data processing failed!")
                    with st.expander("View Error"):
                        st.code(stderr)

    with col2:
        st.write("**Model Training Pipeline**")
        if st.button("🧠 Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                success, stdout, stderr = dashboard.run_model_training()

                if success:
                    st.success("✅ Model training completed successfully!")
                    with st.expander("View Output"):
                        st.code(stdout)
                else:
                    st.error("❌ Model training failed!")
                    with st.expander("View Error"):
                        st.code(stderr)

    # Pipeline Status
    st.subheader("Pipeline Status")

    # Check if processed data exists
    processed_data_path = (
        dashboard.project_root / "data" / "processed" / "processed_movies.parquet"
    )
    model_path = (
        dashboard.project_root / "models" / "content_based" / "similarity_matrix.npy"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if processed_data_path.exists():
            st.success("📊 Processed Data: Available")
            modified_time = datetime.fromtimestamp(processed_data_path.stat().st_mtime)
            st.info(f"Last updated: {modified_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.warning("📊 Processed Data: Missing")

    with col2:
        if model_path.exists():
            st.success("🧠 Trained Model: Available")
            modified_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            st.info(f"Last updated: {modified_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.warning("🧠 Trained Model: Missing")

    with col3:
        api_healthy, _ = dashboard.check_api_status()
        if api_healthy:
            st.success("🚀 API Service: Running")
        else:
            st.error("🚀 API Service: Down")

    # Scheduled Jobs
    st.subheader("Scheduled Jobs")

    st.info("📅 Configure Airflow DAGs for automated pipeline execution")

    # Mock schedule status
    schedules = [
        {"name": "Daily Data Sync", "status": "Active", "next_run": "2025-06-30 12:00"},
        {
            "name": "Weekly Model Retrain",
            "status": "Active",
            "next_run": "2025-07-06 02:00",
        },
        {
            "name": "Hourly Health Check",
            "status": "Active",
            "next_run": "2025-06-30 06:00",
        },
    ]

    df_schedules = pd.DataFrame(schedules)
    st.dataframe(df_schedules, use_container_width=True)


def show_testing(dashboard):
    st.header("API Testing")

    # Recommendation Testing
    st.subheader("Recommendation Testing")

    col1, col2 = st.columns(2)

    with col1:
        movie_id = st.number_input("Movie ID", value=278, min_value=1)
        st.slider("Number of recommendations", 1, 20, 5)

        if st.button("Test Recommendations"):
            with st.spinner("Getting recommendations..."):
                success, result = dashboard.test_recommendation(movie_id)

                if success:
                    st.success(f"✅ Response time: {result['response_time_ms']:.1f}ms")

                    # Display recommendations
                    recs = result["recommendations"]
                    if recs:
                        st.write("**Recommendations:**")
                        for i, rec in enumerate(recs, 1):
                            st.write(
                                f"{i}. {rec['title']} (Similarity: {rec['similarity']:.3f})"
                            )
                    else:
                        st.warning("No recommendations found")
                else:
                    st.error(f"❌ {result['error']}")

    with col2:
        st.subheader("Search Testing")

        query = st.text_input("Search query", value="action")
        st.slider("Search limit", 1, 10, 3)

        if st.button("Test Search"):
            with st.spinner("Searching..."):
                success, result = dashboard.test_search(query)

                if success:
                    st.success(f"✅ Response time: {result['response_time_ms']:.1f}ms")

                    # Display results
                    results = result["results"]
                    if results:
                        st.write("**Search Results:**")
                        for i, movie in enumerate(results, 1):
                            st.write(
                                f"{i}. {movie['title']} (Similarity: {movie['similarity']:.3f})"
                            )
                    else:
                        st.warning("No results found")
                else:
                    st.error(f"❌ {result['error']}")

    # Batch Testing
    st.subheader("Batch Testing")

    if st.button("Run Comprehensive Tests"):
        with st.spinner("Running comprehensive tests..."):
            # Test multiple movies
            test_movies = [278, 238, 240, 13, 19404]
            results = []

            for movie_id in test_movies:
                success, result = dashboard.test_recommendation(movie_id)
                results.append(
                    {
                        "movie_id": movie_id,
                        "success": success,
                        "response_time": (
                            result.get("response_time_ms", 0) if success else 0
                        ),
                        "num_recommendations": (
                            len(result.get("recommendations", [])) if success else 0
                        ),
                    }
                )

            # Display results
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            # Summary
            success_rate = df_results["success"].mean()
            avg_response_time = df_results[df_results["success"]][
                "response_time"
            ].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Success Rate", f"{success_rate:.1%}")
            with col2:
                st.metric("Avg Response Time", f"{avg_response_time:.1f}ms")
            with col3:
                avg_recs = df_results[df_results["success"]][
                    "num_recommendations"
                ].mean()
                st.metric("Avg Recommendations", f"{avg_recs:.1f}")


def show_analytics(dashboard):
    st.header("Performance Analytics")

    # Generate sample analytics data
    st.subheader("Request Analytics")

    # Mock data for demonstration
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D"
    )
    requests_per_day = np.random.poisson(100, len(dates))
    avg_response_time = np.random.normal(50, 10, len(dates))

    df_analytics = pd.DataFrame(
        {
            "date": dates,
            "requests": requests_per_day,
            "response_time": avg_response_time,
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(df_analytics, x="date", y="requests", title="Daily Requests")
        fig.update_layout(yaxis_title="Number of Requests")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            df_analytics, x="date", y="response_time", title="Average Response Time"
        )
        fig.update_layout(yaxis_title="Response Time (ms)")
        st.plotly_chart(fig, use_container_width=True)

    # Model Performance Analytics
    st.subheader("Model Performance")

    # Mock model metrics
    model_metrics = {
        "Precision@5": 0.85,
        "Recall@5": 0.72,
        "NDCG@5": 0.89,
        "Coverage": 0.65,
    }

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precision@5", f"{model_metrics['Precision@5']:.2f}")
    with col2:
        st.metric("Recall@5", f"{model_metrics['Recall@5']:.2f}")
    with col3:
        st.metric("NDCG@5", f"{model_metrics['NDCG@5']:.2f}")
    with col4:
        st.metric("Coverage", f"{model_metrics['Coverage']:.2f}")

    # Recommendation Distribution
    st.subheader("Recommendation Distribution")

    # Mock genre distribution
    genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]
    recommendation_counts = np.random.randint(50, 200, len(genres))

    fig = px.bar(x=genres, y=recommendation_counts, title="Recommendations by Genre")
    fig.update_layout(xaxis_title="Genre", yaxis_title="Number of Recommendations")
    st.plotly_chart(fig, use_container_width=True)


def show_system_metrics(dashboard):
    st.header("System Metrics")

    # Real-time system metrics
    system_metrics = dashboard.get_system_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        (
            "red"
            if system_metrics["cpu_percent"] > 80
            else "orange" if system_metrics["cpu_percent"] > 60 else "green"
        )
        st.metric("CPU Usage", f"{system_metrics['cpu_percent']:.1f}%")
        st.progress(system_metrics["cpu_percent"] / 100)

    with col2:
        (
            "red"
            if system_metrics["memory_percent"] > 80
            else "orange" if system_metrics["memory_percent"] > 60 else "green"
        )
        st.metric("Memory Usage", f"{system_metrics['memory_percent']:.1f}%")
        st.progress(system_metrics["memory_percent"] / 100)

    with col3:
        (
            "red"
            if system_metrics["disk_percent"] > 90
            else "orange" if system_metrics["disk_percent"] > 70 else "green"
        )
        st.metric("Disk Usage", f"{system_metrics['disk_percent']:.1f}%")
        st.progress(system_metrics["disk_percent"] / 100)

    with col4:
        st.metric("Load Average", f"{system_metrics['load_average']:.2f}")

    # Process Information
    st.subheader("Process Information")

    try:
        processes = []
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent"]
        ):
            if "python" in proc.info["name"].lower():
                processes.append(proc.info)

        if processes:
            df_processes = pd.DataFrame(processes)
            st.dataframe(df_processes, use_container_width=True)
        else:
            st.info("No Python processes found")
    except Exception as e:
        st.error(f"Error getting process information: {e}")

    # Disk Space Details
    st.subheader("Disk Space")

    try:
        disk_usage = psutil.disk_usage("/")
        total_gb = disk_usage.total / (1024**3)
        used_gb = disk_usage.used / (1024**3)
        free_gb = disk_usage.free / (1024**3)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Space", f"{total_gb:.1f} GB")
        with col2:
            st.metric("Used Space", f"{used_gb:.1f} GB")
        with col3:
            st.metric("Free Space", f"{free_gb:.1f} GB")

        # Disk usage chart
        fig = go.Figure(
            data=[go.Pie(labels=["Used", "Free"], values=[used_gb, free_gb], hole=0.3)]
        )
        fig.update_layout(title="Disk Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error getting disk information: {e}")


if __name__ == "__main__":
    main()
