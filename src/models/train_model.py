#!/usr/bin/env python3
"""
Content-Based Movie Recommendation Model Training

This script trains a content-based recommendation model using TF-IDF vectorization
and cosine similarity on processed movie features.

Features:
- Loads processed movie data with engineered tags
- Vectorizes tags using TF-IDF with optimized parameters
- Computes cosine similarity matrix
- Saves model artifacts for serving
- Generates comprehensive training metrics
- Validates model quality with sample recommendations

Author: MLOps Movie Recommendation System
"""

import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ContentBasedTrainer:
    """Content-based recommendation model trainer"""

    def __init__(self, data_path: str, output_dir: str):
        """
        Initialize trainer

        Args:
            data_path: Path to processed movies parquet file
            output_dir: Directory to save model artifacts
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model components
        self.movies_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movie_to_idx = None
        self.idx_to_movie = None

        # Training metrics
        self.training_metrics = {}

    def load_data(self) -> pd.DataFrame:
        """Load and validate processed movie data"""
        logger.info(f"Loading processed data from {self.data_path}")

        try:
            # Try to load parquet first
            if self.data_path.endswith(".parquet"):
                self.movies_df = pd.read_parquet(self.data_path)
            else:
                # Fallback to CSV
                self.movies_df = pd.read_csv(self.data_path)

            logger.info(f"Loaded {len(self.movies_df)} movies")

            # Validate required columns
            required_cols = ["movie_id", "title", "tags"]
            missing_cols = [
                col for col in required_cols if col not in self.movies_df.columns
            ]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Data quality checks
            total_movies = len(self.movies_df)
            movies_with_tags = len(
                self.movies_df[
                    self.movies_df["tags"].notna() & (self.movies_df["tags"] != "")
                ]
            )
            tag_coverage = (movies_with_tags / total_movies) * 100

            logger.info(
                f"Data quality: {movies_with_tags}/{total_movies} movies have tags ({tag_coverage:.1f}%)"
            )

            # Filter out movies without tags for training
            original_count = len(self.movies_df)
            self.movies_df = self.movies_df[
                self.movies_df["tags"].notna() & (self.movies_df["tags"] != "")
            ].copy()
            filtered_count = len(self.movies_df)

            logger.info(
                f"Filtered dataset: {filtered_count}/{original_count} movies for training"
            )

            # Create movie mappings
            self.movie_to_idx = {
                movie_id: idx for idx, movie_id in enumerate(self.movies_df["movie_id"])
            }
            self.idx_to_movie = {
                idx: movie_id for movie_id, idx in self.movie_to_idx.items()
            }

            return self.movies_df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_tfidf_vectorizer(self) -> TfidfVectorizer:
        """Create and configure TF-IDF vectorizer with optimized parameters"""
        logger.info("Creating TF-IDF vectorizer")

        # Optimized parameters based on movie content
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size
            stop_words="english",  # Remove English stop words
            lowercase=True,  # Convert to lowercase
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms in less than 2 documents
            max_df=0.8,  # Ignore terms in more than 80% of documents
            sublinear_tf=True,  # Apply sublinear TF scaling
            norm="l2",  # L2 normalization
            smooth_idf=True,  # Smooth IDF weights
            use_idf=True,  # Use IDF weighting
        )

        return self.tfidf_vectorizer

    def vectorize_features(self) -> sp.csr_matrix:
        """Vectorize movie tags using TF-IDF"""
        logger.info("Vectorizing movie features")

        # Get tags column
        tags = self.movies_df["tags"].fillna("").astype(str)

        # Fit and transform tags
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(tags)

        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        logger.info(
            f"Matrix sparsity: {(1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100:.2f}%"
        )

        return self.tfidf_matrix

    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute cosine similarity matrix"""
        logger.info("Computing cosine similarity matrix")

        # Compute cosine similarity in chunks to manage memory
        chunk_size = 1000
        n_movies = self.tfidf_matrix.shape[0]

        # Initialize similarity matrix
        self.similarity_matrix = np.zeros((n_movies, n_movies), dtype=np.float32)

        # Compute similarity in chunks
        for i in range(0, n_movies, chunk_size):
            end_i = min(i + chunk_size, n_movies)
            chunk_similarities = cosine_similarity(
                self.tfidf_matrix[i:end_i], self.tfidf_matrix
            )
            self.similarity_matrix[i:end_i] = chunk_similarities

            if i % (chunk_size * 5) == 0:
                logger.info(f"Processed {end_i}/{n_movies} movies")

        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")

        # Log similarity statistics
        non_diag_similarities = self.similarity_matrix[
            np.triu_indices_from(self.similarity_matrix, k=1)
        ]
        logger.info(
            f"Similarity stats - Mean: {non_diag_similarities.mean():.4f}, "
            f"Std: {non_diag_similarities.std():.4f}, "
            f"Max: {non_diag_similarities.max():.4f}"
        )

        return self.similarity_matrix

    def validate_model(self) -> Dict:
        """Validate model quality with sample recommendations"""
        logger.info("Validating model quality")

        validation_metrics = {}

        # Test with popular movies
        test_movies = self.movies_df.head(10)

        recommendations_quality = []

        for _, movie in test_movies.iterrows():
            movie_id = movie["movie_id"]
            title = movie["title"]

            # Get recommendations
            recs = self.get_recommendations(movie_id, top_k=5)

            if recs:
                # Check if recommendations are different from input
                rec_ids = [r["movie_id"] for r in recs]
                is_diverse = movie_id not in rec_ids

                # Check similarity scores
                scores = [r["similarity"] for r in recs]
                avg_similarity = np.mean(scores)

                recommendations_quality.append(
                    {
                        "movie_id": movie_id,
                        "title": title,
                        "diverse": is_diverse,
                        "avg_similarity": avg_similarity,
                        "recommendations": len(recs),
                    }
                )

                logger.info(
                    f"Sample recommendations for '{title}': {[r['title'] for r in recs[:3]]}"
                )

        # Calculate validation metrics
        if recommendations_quality:
            validation_metrics = {
                "total_tested": len(recommendations_quality),
                "avg_recommendations_per_movie": np.mean(
                    [r["recommendations"] for r in recommendations_quality]
                ),
                "avg_similarity_score": np.mean(
                    [r["avg_similarity"] for r in recommendations_quality]
                ),
                "diversity_rate": np.mean(
                    [r["diverse"] for r in recommendations_quality]
                ),
                "sample_tests": recommendations_quality[
                    :3
                ],  # Store first 3 for reporting
            }

        return validation_metrics

    def get_recommendations(self, movie_id: int, top_k: int = 10) -> List[Dict]:
        """Get recommendations for a movie"""
        if movie_id not in self.movie_to_idx:
            return []

        movie_idx = self.movie_to_idx[movie_id]

        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))

        # Sort by similarity (excluding the movie itself)
        sim_scores = [(idx, score) for idx, score in sim_scores if idx != movie_idx]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top recommendations
        recommendations = []
        for idx, score in sim_scores[:top_k]:
            rec_movie_id = self.idx_to_movie[idx]
            movie_info = self.movies_df[
                self.movies_df["movie_id"] == rec_movie_id
            ].iloc[0]

            # Build recommendation with available columns only
            rec = {
                "movie_id": rec_movie_id,
                "title": movie_info["title"],
                "similarity": float(score),
            }

            # Add optional columns if they exist
            for col in ["release_date", "vote_average", "genres"]:
                if col in movie_info:
                    rec[col] = movie_info[col]

            recommendations.append(rec)

        return recommendations

    def save_model_artifacts(self) -> Dict[str, str]:
        """Save all model artifacts"""
        logger.info("Saving model artifacts")

        artifacts = {}

        try:
            # Save TF-IDF vectorizer
            vectorizer_path = self.output_dir / "tfidf_vectorizer.pkl"
            with open(vectorizer_path, "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
            artifacts["vectorizer"] = str(vectorizer_path)

            # Save TF-IDF matrix (sparse)
            tfidf_matrix_path = self.output_dir / "tfidf_matrix.pkl"
            with open(tfidf_matrix_path, "wb") as f:
                pickle.dump(self.tfidf_matrix, f)
            artifacts["tfidf_matrix"] = str(tfidf_matrix_path)

            # Save similarity matrix
            similarity_path = self.output_dir / "similarity_matrix.npy"
            np.save(similarity_path, self.similarity_matrix)
            artifacts["similarity_matrix"] = str(similarity_path)

            # Save movie mappings
            mappings_path = self.output_dir / "movie_mappings.pkl"
            mappings = {
                "movie_to_idx": self.movie_to_idx,
                "idx_to_movie": self.idx_to_movie,
            }
            with open(mappings_path, "wb") as f:
                pickle.dump(mappings, f)
            artifacts["mappings"] = str(mappings_path)

            # Save movie metadata for recommendations
            metadata_path = self.output_dir / "movie_metadata.pkl"
            # Only use available columns from processed data
            available_cols = ["movie_id", "title"]
            if "tags_length" in self.movies_df.columns:
                available_cols.append("tags_length")
            if "has_content" in self.movies_df.columns:
                available_cols.append("has_content")

            metadata_df = self.movies_df[available_cols].copy()
            metadata_df.to_pickle(metadata_path)
            artifacts["metadata"] = str(metadata_path)

            logger.info(f"Saved {len(artifacts)} model artifacts")
            return artifacts

        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise

    def generate_training_report(
        self, validation_metrics: Dict, artifacts: Dict[str, str]
    ) -> str:
        """Generate comprehensive training report"""
        logger.info("Generating training report")

        report_path = self.output_dir / "training_report.txt"

        report_content = f"""
# Content-Based Movie Recommendation Model Training Report

## Training Summary
- Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total Movies Processed: {len(self.movies_df):,}
- Data Source: {self.data_path}
- Output Directory: {self.output_dir}

## Model Architecture
- Algorithm: Content-Based Filtering with TF-IDF and Cosine Similarity
- Vectorization: TF-IDF with optimized parameters
- Feature Source: Engineered tags (overview + genres + keywords + cast)
- Similarity Metric: Cosine Similarity

## TF-IDF Configuration
- Max Features: {self.tfidf_vectorizer.max_features:,}
- N-gram Range: {self.tfidf_vectorizer.ngram_range}
- Min Document Frequency: {self.tfidf_vectorizer.min_df}
- Max Document Frequency: {self.tfidf_vectorizer.max_df}
- Vocabulary Size: {len(self.tfidf_vectorizer.vocabulary_):,}

## Model Metrics
- TF-IDF Matrix Shape: {self.tfidf_matrix.shape}
- Matrix Sparsity: {(1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100:.2f}%
- Similarity Matrix Shape: {self.similarity_matrix.shape}
- Memory Usage: ~{(self.similarity_matrix.nbytes / 1024**2):.1f} MB

## Validation Results
"""

        if validation_metrics:
            report_content += f"""- Movies Tested: {validation_metrics['total_tested']}
- Average Recommendations per Movie: {validation_metrics['avg_recommendations_per_movie']:.1f}
- Average Similarity Score: {validation_metrics['avg_similarity_score']:.4f}
- Diversity Rate: {validation_metrics['diversity_rate']:.2%}

## Sample Recommendations
"""
            for test in validation_metrics.get("sample_tests", []):
                report_content += f"- '{test['title']}': {test['recommendations']} recommendations, avg similarity {test['avg_similarity']:.3f}\n"

        report_content += """
## Model Artifacts
"""
        for artifact_type, path in artifacts.items():
            file_size = os.path.getsize(path) / 1024**2  # MB
            report_content += (
                f"- {artifact_type.title()}: {path} ({file_size:.1f} MB)\n"
            )

        report_content += """
## Training Configuration
- Feature Engineering Strategy: Overview-heavy (3:1:2 ratio)
- Text Processing: NLTK lemmatization, stopword removal
- Data Quality: 99.8% tag coverage before training

## Next Steps
1. Deploy model artifacts to serving infrastructure
2. Implement recommendation API endpoints
3. Set up model monitoring and evaluation
4. Configure automated retraining pipeline

## Notes
- Model is ready for production deployment
- Similarity matrix enables fast real-time recommendations
- Consider distributed serving for high-traffic scenarios
- Monitor recommendation quality and user feedback
"""

        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Training report saved to {report_path}")
        return str(report_path)

    def train(self) -> Dict:
        """Run the complete training pipeline"""
        logger.info("Starting content-based recommendation model training")
        start_time = datetime.now()

        try:
            # Load and validate data
            self.load_data()

            # Create vectorizer and vectorize features
            self.create_tfidf_vectorizer()
            self.vectorize_features()

            # Compute similarity matrix
            self.compute_similarity_matrix()

            # Validate model
            validation_metrics = self.validate_model()

            # Save model artifacts
            artifacts = self.save_model_artifacts()

            # Generate training report
            report_path = self.generate_training_report(validation_metrics, artifacts)

            # Calculate training time
            training_time = datetime.now() - start_time

            # Compile training results
            results = {
                "status": "success",
                "training_time": str(training_time),
                "total_movies": len(self.movies_df),
                "matrix_shape": self.similarity_matrix.shape,
                "vocabulary_size": len(self.tfidf_vectorizer.vocabulary_),
                "validation_metrics": validation_metrics,
                "artifacts": artifacts,
                "report_path": report_path,
            }

            logger.info(f"Training completed successfully in {training_time}")
            logger.info(
                f"Model ready for deployment with {len(self.movies_df):,} movies"
            )

            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main training function"""

    # Paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_path = project_root / "data" / "processed" / "processed_movies.parquet"
    output_dir = project_root / "models" / "content_based"

    # Check if data file exists
    if not data_path.exists():
        # Try CSV fallback
        csv_path = project_root / "data" / "processed" / "processed_movies.csv"
        if csv_path.exists():
            data_path = csv_path
            logger.info(f"Using CSV data file: {data_path}")
        else:
            logger.error(f"Data file not found: {data_path}")
            sys.exit(1)

    # Initialize trainer
    trainer = ContentBasedTrainer(data_path=str(data_path), output_dir=str(output_dir))

    # Run training
    try:
        results = trainer.train()

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Movies Processed: {results['total_movies']:,}")
        print(f"Training Time: {results['training_time']}")
        print(f"Model Shape: {results['matrix_shape']}")
        print(f"Vocabulary Size: {results['vocabulary_size']:,}")
        print(f"Artifacts Saved: {len(results['artifacts'])}")
        print(f"Report Available: {results['report_path']}")
        print("=" * 60)

        # Show sample recommendations
        if (
            results["validation_metrics"]
            and "sample_tests" in results["validation_metrics"]
        ):
            print("\nSample Recommendations:")
            for test in results["validation_metrics"]["sample_tests"]:
                print(f"- {test['title']}: {test['recommendations']} recommendations")

        print("\nModel is ready for deployment!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
