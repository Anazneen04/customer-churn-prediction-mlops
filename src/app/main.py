"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    # Demographics
    gender: str                # "Male" or "Female"
    Partner: str               # "Yes" or "No" - has partner
    Dependents: str            # "Yes" or "No" - has dependents
    
    # Phone services
    PhoneService: str          # "Yes" or "No"
    MultipleLines: str         # "Yes", "No", or "No phone service"
    
    # Internet services  
    InternetService: str       # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    OnlineBackup: str          # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    StreamingTV: str           # "Yes", "No", or "No internet service"
    StreamingMovies: str       # "Yes", "No", or "No internet service"
    
    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check", etc.
    
    # Numeric features
    tenure: int                # Number of months with company
    MonthlyCharges: float      # Monthly charges in dollars
    TotalCharges: float        # Total charges to date

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.
    
    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


# =================================================== # 

# === GRADIO UI CONFIGURATION ===

def process_prediction(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """
    Adapter function mapping Gradio inputs to the inference pipeline.
    Returns styled HTML blocks for professional dashboard rendering.
    """
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    
    try:
        result = predict(data)
        
        # Professional HTML Output formatting
        if result == "Likely to churn":
            return f"""
            <div style="background-color: #fee2e2; border-left: 6px solid #ef4444; padding: 20px; border-radius: 8px; font-family: sans-serif;">
                <h3 style="color: #991b1b; margin-top: 0; display: flex; align-items: center; font-size: 1.5rem;">
                    <span style="font-size: 1.8rem; margin-right: 10px;">⚠️</span> High Risk
                </h3>
                <p style="color: #7f1d1d; font-size: 1.1rem; margin-bottom: 0;">
                    <strong>Alert:</strong> This customer is highly likely to churn. Immediate retention intervention is recommended.
                </p>
            </div>
            """
        else:
            return f"""
            <div style="background-color: #dcfce7; border-left: 6px solid #22c55e; padding: 20px; border-radius: 8px; font-family: sans-serif;">
                <h3 style="color: #166534; margin-top: 0; display: flex; align-items: center; font-size: 1.5rem;">
                    <span style="font-size: 1.8rem; margin-right: 10px;">✅</span> Low Risk
                </h3>
                <p style="color: #14532d; font-size: 1.1rem; margin-bottom: 0;">
                    <strong>Safe:</strong> This customer is expected to stay. No immediate action required.
                </p>
            </div>
            """
    except Exception as e:
        return f"""
        <div style="background-color: #f3f4f6; border-left: 6px solid #6b7280; padding: 20px; border-radius: 8px; font-family: sans-serif;">
            <h3 style="color: #374151; margin-top: 0;">Error Processing Request</h3>
            <p style="color: #4b5563;">{str(e)}</p>
        </div>
        """

# Define the custom theme for professional look
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="*shadow_sm"
)

with gr.Blocks(theme=custom_theme, title="Customer Churn Prediction", css="footer {visibility: hidden}") as demo:
    
    # Application Header
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # Customer Churn Prediction
                ### Identify at-risk customers to improve retention workflows
                Enter customer details below or select an example to evaluate their churn risk probability. 
                Our model utilizes historical data and XGBoost to deliver real-time risk assessments.
                """
            )
            
    with gr.Row():
        
        # Left Column - Inputs (70% width theoretically handled by scale=7/3)
        with gr.Column(scale=7):
            
            with gr.Tabs():
                # TAB 1: Demographics
                with gr.TabItem("👤 Demographics"):
                    with gr.Row():
                        gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                        partner = gr.Radio(["Yes", "No"], label="Partner", value="No")
                        dependents = gr.Radio(["Yes", "No"], label="Dependents", value="No")
                
                # TAB 2: Services
                with gr.TabItem("⚙️ Services"):
                    with gr.Row():
                        phone_service = gr.Radio(["Yes", "No"], label="Phone Service", value="Yes")
                        multiple_lines = gr.Radio(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
                    
                    gr.Markdown("#### Internet Add-ons")
                    with gr.Group():
                        internet_service = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                        with gr.Row():
                            online_security = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                            online_backup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                            device_protection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                        with gr.Row():
                            tech_support = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                            streaming_tv = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
                            streaming_movies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")
                            
                # TAB 3: Account & Billing
                with gr.TabItem("💳 Account & Billing"):
                    with gr.Row():
                        contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                        paperless_billing = gr.Radio(["Yes", "No"], label="Paperless Billing", value="Yes")
                        payment_method = gr.Dropdown([
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"
                        ], label="Payment Method", value="Electronic check")
                    
                    with gr.Row():
                        tenure = gr.Slider(minimum=0, maximum=100, value=1, step=1, label="Tenure (months)", info="Number of months customer has stayed with the company")
                        monthly_charges = gr.Number(label="Monthly Charges ($)", value=85.0)
                        total_charges = gr.Number(label="Total Charges ($)", value=85.0, info="Lifetime value of customer")
        
        # Right Column - Output & Actions
        with gr.Column(scale=3):
            # Prediction Output Card
            prediction_output = gr.HTML(
                value='''
                <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; padding: 20px; border-radius: 8px; text-align: center; color: #64748b; font-family: sans-serif;">
                    Enter customer details and click <strong>Predict Churn Risk</strong> to see the result.
                </div>
                ''',
                label="Risk Assessment"
            )
            
            with gr.Row():
                predict_btn = gr.Button("Predict Churn Risk", variant="primary", size="lg")
            with gr.Row():
                clear_btn = gr.Button("Clear Form", variant="secondary")

            gr.Markdown("### 📌 Quick Examples")
            gr.Examples(
                examples=[
                    # High risk
                    ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 1, 85.0, 85.0],
                    # Low risk
                    ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No", "Two year", "No", "Credit card (automatic)", 60, 45.0, 2700.0]
                ],
                inputs=[
                    gender, partner, dependents, phone_service, multiple_lines,
                    internet_service, online_security, online_backup, device_protection,
                    tech_support, streaming_tv, streaming_movies, contract,
                    paperless_billing, payment_method, tenure, monthly_charges, total_charges
                ],
                label=""
            )

    # Wire up the button click
    input_components = [
        gender, partner, dependents, phone_service, multiple_lines,
        internet_service, online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies, contract,
        paperless_billing, payment_method, tenure, monthly_charges, total_charges
    ]
    
    predict_btn.click(
        fn=process_prediction,
        inputs=input_components,
        outputs=prediction_output
    )
    
    # Wire up the clear button
    def clear_form():
        return (
            "Male", "No", "No", "Yes", "No", "Fiber optic", 
            "No", "No", "No", "No", "Yes", "Yes", 
            "Month-to-month", "Yes", "Electronic check", 
            1, 85.0, 85.0,
            '''
            <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; padding: 20px; border-radius: 8px; text-align: center; color: #64748b; font-family: sans-serif;">
                Enter customer details and click <strong>Predict Churn Risk</strong> to see the result.
            </div>
            '''
        )
        
    clear_btn.click(
        fn=clear_form,
        inputs=[],
        outputs=input_components + [prediction_output]
    )

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)