#!/usr/bin/env python3
"""
Model Performance Monitoring

Monitors recommendation model performance and generates alerts
when quality metrics fall below thresholds.

Features:
- API response time monitoring
- Recommendation quality checks
- Model drift detection
- Performance alerting
- Metrics collection for dashboard

Author: MLOps Movie Recommendation System
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Model performance monitoring system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.metrics_file = Path("monitoring/model_metrics.json")
        self.metrics_file.parent.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.thresholds = {
            'response_time_ms': 1000,
            'min_recommendations': 3,
            'min_similarity': 0.05,
            'api_success_rate': 0.95
        }
        
        self.test_movie_ids = [278, 238, 240, 13, 19404]  # Popular movies for testing
    
    def check_api_health(self) -> Dict:
        """Check API health and response time"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'healthy',
                    'response_time_ms': response_time,
                    'api_data': data
                }
            else:
                return {
                    'status': 'unhealthy',
                    'response_time_ms': response_time,
                    'error': f"Status code: {response.status_code}"
                }
        except Exception as e:
            return {
                'status': 'error',
                'response_time_ms': float('inf'),
                'error': str(e)
            }
    
    def test_recommendations(self) -> Dict:
        """Test recommendation quality"""
        results = []
        
        for movie_id in self.test_movie_ids:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/recommendations",
                    json={"movie_id": movie_id, "top_k": 5},
                    timeout=10
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get('recommendations', [])
                    
                    # Quality metrics
                    num_recs = len(recommendations)
                    avg_similarity = np.mean([r['similarity'] for r in recommendations]) if recommendations else 0
                    
                    results.append({
                        'movie_id': movie_id,
                        'status': 'success',
                        'response_time_ms': response_time,
                        'num_recommendations': num_recs,
                        'avg_similarity': avg_similarity
                    })
                else:
                    results.append({
                        'movie_id': movie_id,
                        'status': 'error',
                        'response_time_ms': response_time,
                        'error': f"Status code: {response.status_code}"
                    })
                    
            except Exception as e:
                results.append({
                    'movie_id': movie_id,
                    'status': 'error',
                    'response_time_ms': float('inf'),
                    'error': str(e)
                })
        
        # Aggregate metrics
        successful_tests = [r for r in results if r['status'] == 'success']
        success_rate = len(successful_tests) / len(results)
        
        if successful_tests:
            avg_response_time = np.mean([r['response_time_ms'] for r in successful_tests])
            avg_num_recs = np.mean([r['num_recommendations'] for r in successful_tests])
            avg_similarity = np.mean([r['avg_similarity'] for r in successful_tests])
        else:
            avg_response_time = float('inf')
            avg_num_recs = 0
            avg_similarity = 0
        
        return {
            'success_rate': success_rate,
            'avg_response_time_ms': avg_response_time,
            'avg_num_recommendations': avg_num_recs,
            'avg_similarity': avg_similarity,
            'individual_results': results
        }
    
    def test_search_functionality(self) -> Dict:
        """Test search functionality"""
        test_queries = ["action", "comedy", "drama", "godfather", "matrix"]
        results = []
        
        for query in test_queries:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/search",
                    json={"query": query, "top_k": 3},
                    timeout=10
                )
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    search_results = data.get('results', [])
                    
                    results.append({
                        'query': query,
                        'status': 'success',
                        'response_time_ms': response_time,
                        'num_results': len(search_results)
                    })
                else:
                    results.append({
                        'query': query,
                        'status': 'error',
                        'response_time_ms': response_time,
                        'error': f"Status code: {response.status_code}"
                    })
                    
            except Exception as e:
                results.append({
                    'query': query,
                    'status': 'error',
                    'response_time_ms': float('inf'),
                    'error': str(e)
                })
        
        # Aggregate metrics
        successful_searches = [r for r in results if r['status'] == 'success']
        search_success_rate = len(successful_searches) / len(results)
        
        if successful_searches:
            avg_search_time = np.mean([r['response_time_ms'] for r in successful_searches])
            avg_results = np.mean([r['num_results'] for r in successful_searches])
        else:
            avg_search_time = float('inf')
            avg_results = 0
        
        return {
            'search_success_rate': search_success_rate,
            'avg_search_time_ms': avg_search_time,
            'avg_num_results': avg_results,
            'individual_results': results
        }
    
    def run_monitoring_cycle(self) -> Dict:
        """Run complete monitoring cycle"""
        logger.info("Starting monitoring cycle")
        
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'health_check': self.check_api_health(),
            'recommendations_test': self.test_recommendations(),
            'search_test': self.test_search_functionality()
        }
        
        # Check for threshold violations
        alerts = self.check_thresholds(monitoring_results)
        monitoring_results['alerts'] = alerts
        
        # Save metrics
        self.save_metrics(monitoring_results)
        
        # Log summary
        self.log_summary(monitoring_results)
        
        return monitoring_results
    
    def check_thresholds(self, results: Dict) -> List[Dict]:
        """Check if any metrics violate thresholds"""
        alerts = []
        
        # API Health checks
        health = results['health_check']
        if health['status'] != 'healthy':
            alerts.append({
                'type': 'API_DOWN',
                'message': f"API is unhealthy: {health.get('error', 'Unknown error')}",
                'severity': 'CRITICAL'
            })
        elif health['response_time_ms'] > self.thresholds['response_time_ms']:
            alerts.append({
                'type': 'SLOW_RESPONSE',
                'message': f"API response time {health['response_time_ms']:.1f}ms exceeds threshold {self.thresholds['response_time_ms']}ms",
                'severity': 'WARNING'
            })
        
        # Recommendation quality checks
        rec_test = results['recommendations_test']
        if rec_test['success_rate'] < self.thresholds['api_success_rate']:
            alerts.append({
                'type': 'LOW_SUCCESS_RATE',
                'message': f"Recommendation success rate {rec_test['success_rate']:.2%} below threshold {self.thresholds['api_success_rate']:.2%}",
                'severity': 'CRITICAL'
            })
        
        if rec_test['avg_num_recommendations'] < self.thresholds['min_recommendations']:
            alerts.append({
                'type': 'LOW_RECOMMENDATION_COUNT',
                'message': f"Average recommendations {rec_test['avg_num_recommendations']:.1f} below threshold {self.thresholds['min_recommendations']}",
                'severity': 'WARNING'
            })
        
        if rec_test['avg_similarity'] < self.thresholds['min_similarity']:
            alerts.append({
                'type': 'LOW_SIMILARITY',
                'message': f"Average similarity {rec_test['avg_similarity']:.3f} below threshold {self.thresholds['min_similarity']}",
                'severity': 'WARNING'
            })
        
        return alerts
    
    def save_metrics(self, results: Dict):
        """Save metrics to file"""
        try:
            # Load existing metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            # Add new metrics
            all_metrics.append(results)
            
            # Keep only last 1000 entries
            all_metrics = all_metrics[-1000:]
            
            # Save back to file
            with open(self.metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def log_summary(self, results: Dict):
        """Log monitoring summary"""
        health = results['health_check']
        rec_test = results['recommendations_test']
        search_test = results['search_test']
        alerts = results['alerts']
        
        logger.info("=== MONITORING SUMMARY ===")
        logger.info(f"API Health: {health['status']} ({health['response_time_ms']:.1f}ms)")
        logger.info(f"Recommendations: {rec_test['success_rate']:.2%} success, {rec_test['avg_response_time_ms']:.1f}ms avg")
        logger.info(f"Search: {search_test['search_success_rate']:.2%} success, {search_test['avg_search_time_ms']:.1f}ms avg")
        
        if alerts:
            logger.warning(f"ALERTS ({len(alerts)}):")
            for alert in alerts:
                logger.warning(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")
        else:
            logger.info("No alerts detected")
        
        logger.info("=" * 26)
    
    def generate_monitoring_report(self) -> str:
        """Generate detailed monitoring report"""
        if not self.metrics_file.exists():
            return "No monitoring data available"
        
        # Load recent metrics
        with open(self.metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        if not all_metrics:
            return "No monitoring data available"
        
        # Analyze last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_metrics = [
            m for m in all_metrics 
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        if not recent_metrics:
            return "No recent monitoring data available"
        
        # Calculate statistics
        health_data = [m['health_check'] for m in recent_metrics]
        rec_data = [m['recommendations_test'] for m in recent_metrics]
        search_data = [m['search_test'] for m in recent_metrics]
        
        # Generate report
        report = f"""
# Model Monitoring Report (Last 24 Hours)

## Summary
- Total Monitoring Cycles: {len(recent_metrics)}
- Time Range: {recent_metrics[0]['timestamp']} to {recent_metrics[-1]['timestamp']}

## API Health Metrics
- Average Response Time: {np.mean([h['response_time_ms'] for h in health_data if h['response_time_ms'] != float('inf')]):.1f}ms
- Uptime: {len([h for h in health_data if h['status'] == 'healthy']) / len(health_data):.2%}

## Recommendation Performance
- Success Rate: {np.mean([r['success_rate'] for r in rec_data]):.2%}
- Average Response Time: {np.mean([r['avg_response_time_ms'] for r in rec_data if r['avg_response_time_ms'] != float('inf')]):.1f}ms
- Average Recommendations per Request: {np.mean([r['avg_num_recommendations'] for r in rec_data]):.1f}
- Average Similarity Score: {np.mean([r['avg_similarity'] for r in rec_data]):.3f}

## Search Performance
- Success Rate: {np.mean([s['search_success_rate'] for s in search_data]):.2%}
- Average Response Time: {np.mean([s['avg_search_time_ms'] for s in search_data if s['avg_search_time_ms'] != float('inf')]):.1f}ms

## Recent Alerts
"""
        
        # Add recent alerts
        all_alerts = []
        for m in recent_metrics:
            for alert in m.get('alerts', []):
                alert['timestamp'] = m['timestamp']
                all_alerts.append(alert)
        
        if all_alerts:
            for alert in all_alerts[-10:]:  # Last 10 alerts
                report += f"- [{alert['timestamp']}] {alert['severity']}: {alert['message']}\n"
        else:
            report += "- No alerts in the last 24 hours\n"
        
        return report


def main():
    """Main monitoring function"""
    monitor = ModelMonitor()
    
    # Run single monitoring cycle
    results = monitor.run_monitoring_cycle()
    
    # Generate and save report
    report = monitor.generate_monitoring_report()
    report_path = Path("monitoring/monitoring_report.md")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Monitoring report saved to {report_path}")


if __name__ == "__main__":
    main()
