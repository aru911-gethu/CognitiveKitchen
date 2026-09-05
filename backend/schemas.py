from pydantic import BaseModel, Field
from typing import List, Optional

class Ingredient(BaseModel):
    name: str = Field(..., description="Normalized ingredient name (e.g., 'chicken breast')")
    quantity: Optional[float] = Field(None, description="Numeric quantity, optional if missing on site")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., 'grams', 'cups')")

class Recipe(BaseModel):
    id: str = Field(..., description="Unique hash or identifier for the recipe")
    title: str = Field(..., description="Name of the recipe")
    prep_time_minutes: Optional[int] = Field(None, description="Preparation time in minutes")
    dietary_tags: List[str] = Field(default_factory=list, description="e.g., ['vegan', 'gluten-free']")
    ingredients: List[Ingredient] = Field(..., description="Structured list of ingredients")
    instructions: List[str] = Field(..., description="Step-by-step cooking instructions")
    
    class Config:
        frozen = True