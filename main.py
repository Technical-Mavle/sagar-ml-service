# In sagar-ml-service/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware # Import CORS
from pydantic import BaseModel, Field
from config import supabase
import pandas as pd
import geopandas as gpd # For geospatial operations
from shapely.geometry import box # For creating geometries
import io
import numpy as np
import uuid
from typing import Dict, Any

app = FastAPI(
    title="SAGAR Advanced AI/ML Service",
    description="A service for running advanced geospatial and analytical tasks.",
    version="2.1.0"
)

# --- NEW: Add CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory job store (for a real app, use Redis or a database) ---
jobs: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class CorrelationRequest(BaseModel):
    file1_path: str = Field(..., example="occurrence.parquet")
    file2_path: str = Field(..., example="temperature_data.parquet")
    column1: str = Field(..., example="individualCount")
    column2: str = Field(..., example="TO3")
    file1_lat_col: str = Field(default="decimalLatitude", example="decimalLatitude")
    file1_lon_col: str = Field(default="decimalLongitude", example="decimalLongitude")
    file2_lat_col: str = Field(default="lat", example="lat")
    file2_lon_col: str = Field(default="lon", example="lon")

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

def download_and_prepare_gdf(file_path: str, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    """Downloads a parquet file and converts it to a GeoDataFrame."""
    try:
        response = supabase.storage.from_('processed-data').download(file_path)
        df = pd.read_parquet(io.BytesIO(response))

        if not all(col in df.columns for col in [lat_col, lon_col]):
            raise ValueError(f"Missing required coordinate columns '{lat_col}' or '{lon_col}' in file {file_path}")
        
        geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf
    except Exception as e:
        raise RuntimeError(f"Failed to process file {file_path}: {e}")

def run_geospatial_correlation(job_id: str, request: CorrelationRequest):
    """The actual long-running analysis task with projection for accuracy."""
    try:
        jobs[job_id]['status'] = 'running'
        
        gdf1 = download_and_prepare_gdf(request.file1_path, request.file1_lat_col, request.file1_lon_col)
        gdf2 = download_and_prepare_gdf(request.file2_path, request.file2_lat_col, request.file2_lon_col)

        gdf1_projected = gdf1.to_crs("EPSG:3395")
        gdf2_projected = gdf2.to_crs("EPSG:3395")

        joined_gdf = gpd.sjoin_nearest(gdf1_projected, gdf2_projected, max_distance=10000, how="inner")

        if joined_gdf.empty:
            raise ValueError("No matching data points found within 10km after spatial join.")
        
        if not all(col in joined_gdf.columns for col in [request.column1, request.column2]):
            raise ValueError("Correlation columns not found after spatial join.")

        correlation = joined_gdf[[request.column1, request.column2]].corr().iloc[0, 1]
        
        summary_text = f"The geospatial correlation between '{request.column1}' and '{request.column2}' is {correlation:.4f}."
        if abs(correlation) > 0.7:
            summary_text += " This indicates a strong correlation."
        elif abs(correlation) > 0.4:
            summary_text += " This indicates a moderate correlation."
        else:
            summary_text += " This indicates a weak or no correlation."

        plot_df = joined_gdf[[request.column1, request.column2]].dropna().to_dict(orient='records')

        heatmap_df = joined_gdf[[request.column1, request.column2]].dropna()
        x_bins = np.linspace(heatmap_df[request.column1].min(), heatmap_df[request.column1].max(), 10)
        y_bins = np.linspace(heatmap_df[request.column2].min(), heatmap_df[request.column2].max(), 10)
        heatmap, _, _ = np.histogram2d(heatmap_df[request.column1], heatmap_df[request.column2], bins=[x_bins, y_bins])
        
        heatmap_data = {
            "x_labels": [f"{val:.2f}" for val in x_bins],
            "y_labels": [f"{val:.2f}" for val in y_bins],
            "values": heatmap.T.tolist()
        }

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {
            "summary": summary_text,
            "plot_data": plot_df,
            "heatmap_data": heatmap_data,
            "correlation_coefficient": correlation
        }

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

@app.post("/analyze/geospatial-correlation", response_model=JobResponse)
async def create_geospatial_correlation_job(request: CorrelationRequest, background_tasks: BackgroundTasks):
    """Accepts a geospatial correlation job and runs it in the background."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    
    background_tasks.add_task(run_geospatial_correlation, job_id, request)
    
    return {"job_id": job_id, "status": "pending", "message": "Job accepted and is running in the background."}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Retrieves the status or result of a background job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=444, detail="Job not found.") # Changed status to 444 for clarity
    
    if job['status'] == 'completed':
        return job['result']
    elif job['status'] == 'failed':
        return {"status": "failed", "error": job.get('error')}
    else:
        return {"status": job['status']}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "SAGAR Advanced AI/ML Service is running."}