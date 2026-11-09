"""
Pydantic schemas with strict validation for BRFSS features and request/response payloads.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


class BRFSSFeatures(BaseModel):
    """
    BRFSS 21 features with strict ranges based on the training dataset.
    Unknown fields are forbidden.
    All fields are optional because the Feature Assembler handles missing fields.
    """
    model_config = ConfigDict(extra="forbid")

    HighBP: Optional[float] = Field(None, ge=0.0, le=1.0, description="High blood pressure adult (0=no, 1=yes)")
    HighChol: Optional[float] = Field(None, ge=0.0, le=1.0, description="High cholesterol adult (0=no, 1=yes)")
    CholCheck: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cholesterol check within 5 years (0=no, 1=yes)")
    BMI: Optional[float] = Field(None, ge=12.0, le=98.0, description="Body Mass Index (12-98)")
    Smoker: Optional[float] = Field(None, ge=0.0, le=1.0, description="Smoked at least 100 cigarettes (0=no, 1=yes)")
    Stroke: Optional[float] = Field(None, ge=0.0, le=1.0, description="Ever told you had a stroke (0=no, 1=yes)")
    HeartDiseaseorAttack: Optional[float] = Field(None, ge=0.0, le=1.0, description="Coronary heart disease or MI (0=no, 1=yes)")
    PhysActivity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Physical activity in past 30 days (0=no, 1=yes)")
    Fruits: Optional[float] = Field(None, ge=0.0, le=1.0, description="Consume fruit 1 or more times per day (0=no, 1=yes)")
    Veggies: Optional[float] = Field(None, ge=0.0, le=1.0, description="Consume vegetables 1 or more times per day (0=no, 1=yes)")
    HvyAlcoholConsump: Optional[float] = Field(None, ge=0.0, le=1.0, description="Heavy alcohol consumption (0=no, 1=yes)")
    AnyHealthcare: Optional[float] = Field(None, ge=0.0, le=1.0, description="Have any health care coverage (0=no, 1=yes)")
    NoDocbcCost: Optional[float] = Field(None, ge=0.0, le=1.0, description="Could not see doctor because of cost (0=no, 1=yes)")
    GenHlth: Optional[float] = Field(None, ge=1.0, le=5.0, description="General health rating (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor)")
    MentHlth: Optional[float] = Field(None, ge=0.0, le=30.0, description="Days with poor mental health in past 30 days (0-30)")
    PhysHlth: Optional[float] = Field(None, ge=0.0, le=30.0, description="Days with physical illness/injury in past 30 days (0-30)")
    DiffWalk: Optional[float] = Field(None, ge=0.0, le=1.0, description="Serious difficulty walking or climbing stairs (0=no, 1=yes)")
    Sex: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sex (0=female, 1=male)")
    Age: Optional[float] = Field(None, ge=1.0, le=13.0, description="Age category 1-13 (1=18-24 ... 13=80+)")
    Education: Optional[float] = Field(None, ge=1.0, le=6.0, description="Education level 1-6")
    Income: Optional[float] = Field(None, ge=1.0, le=8.0, description="Income level 1-8")


class RiskRequest(BRFSSFeatures):
    pass


class NLPRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, description="Clinical or free-text note for biomedical NER")


class BaseResponse(BaseModel):
    disclaimer: str
    limitations: List[str]


class Reliability(BaseModel):
    level: str = Field(..., description="'low' | 'medium' | 'high'")
    reasons: List[str] = Field(default_factory=list)


class RiskResponse(BaseResponse):
    predicted_class: str
    probabilities: Dict[str, float]
    completeness: str
    assumed_features: List[str]
    provided_or_extracted_count: int
    assumed_count: int
    known_condition: bool
    features_used: int = 21
    reliability: Optional[Reliability] = None


class ExplainResponse(BaseResponse):
    predicted_class: str
    probabilities: Dict[str, float]
    shap_top_features: List[Dict[str, Any]]
    lime_top_features: List[Dict[str, Any]]
    completeness: str
    assumed_features: List[str]

