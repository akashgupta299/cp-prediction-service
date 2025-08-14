"""Database models and operations for prediction history"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import logging
from .config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class PredictionHistory(Base):
    """Database model for storing prediction history"""
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    shipment_id = Column(String(100), index=True, nullable=False)
    address = Column(Text, nullable=False)
    pincode = Column(String(6), index=True, nullable=False)
    predicted_cp = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    cached = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    
    def __repr__(self):
        return f"<PredictionHistory(shipment_id='{self.shipment_id}', predicted_cp='{self.predicted_cp}')>"


class DatabaseManager:
    """Database manager for handling prediction history"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            self.engine = create_engine(
                settings.database_url,
                connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            self.engine = None
            self.SessionLocal = None
    
    def get_session(self) -> Optional[Session]:
        """Get database session"""
        if self.SessionLocal is None:
            return None
        return self.SessionLocal()
    
    def save_prediction(
        self,
        shipment_id: str,
        address: str,
        pincode: str,
        predicted_cp: str,
        confidence: float,
        model_version: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        cached: bool = False
    ) -> bool:
        """Save prediction to database"""
        if not settings.enable_prediction_history or self.SessionLocal is None:
            return False
            
        session = self.get_session()
        if session is None:
            return False
            
        try:
            prediction = PredictionHistory(
                shipment_id=shipment_id,
                address=address,
                pincode=pincode,
                predicted_cp=predicted_cp,
                confidence=confidence,
                model_version=model_version,
                processing_time_ms=processing_time_ms,
                cached=cached
            )
            
            session.add(prediction)
            session.commit()
            logger.debug(f"Saved prediction for shipment: {shipment_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to save prediction: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_prediction_history(
        self, 
        shipment_id: Optional[str] = None, 
        limit: int = 100
    ) -> List[PredictionHistory]:
        """Get prediction history"""
        if self.SessionLocal is None:
            return []
            
        session = self.get_session()
        if session is None:
            return []
            
        try:
            query = session.query(PredictionHistory)
            
            if shipment_id:
                query = query.filter(PredictionHistory.shipment_id == shipment_id)
            
            predictions = query.order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
            return predictions
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get prediction history: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_statistics(self) -> dict:
        """Get prediction statistics"""
        if self.SessionLocal is None:
            return {}
            
        session = self.get_session()
        if session is None:
            return {}
            
        try:
            total_predictions = session.query(PredictionHistory).count()
            avg_confidence = session.query(PredictionHistory.confidence).filter(
                PredictionHistory.confidence.isnot(None)
            ).all()
            
            avg_conf_value = sum(c[0] for c in avg_confidence) / len(avg_confidence) if avg_confidence else 0.0
            
            return {
                "total_predictions": total_predictions,
                "average_confidence": avg_conf_value,
                "cache_hit_rate": 0.0  # Calculate based on cached field if needed
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
        finally:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()
