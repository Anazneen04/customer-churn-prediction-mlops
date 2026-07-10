import pandas as pd
from typing import Tuple, List

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using basic Pandas checks.
    
    This replaces Great Expectations V1 dataset logic since the API has completely changed.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    print("🔍 Starting data validation...")
    
    expectations_passed = True
    failed_expectations = []
    
    def log_failure(msg):
        nonlocal expectations_passed
        expectations_passed = False
        failed_expectations.append(msg)
        
    print("   📋 Validating schema and required columns...")
    
    # Check if required columns exist
    required_cols = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            log_failure(f"Missing column: {col}")
            
    if "customerID" in df.columns and df["customerID"].isnull().any():
        log_failure("customerID contains null values")
        
    print("   💼 Validating business logic constraints...")
    
    if "gender" in df.columns:
        invalid_genders = df[~df["gender"].isin(["Male", "Female"])]
        if not invalid_genders.empty:
            log_failure("Invalid gender values found")
            
    if "Partner" in df.columns:
        invalid_partners = df[~df["Partner"].isin(["Yes", "No"])]
        if not invalid_partners.empty:
            log_failure("Invalid Partner values found")
            
    if "Dependents" in df.columns:
        invalid = df[~df["Dependents"].isin(["Yes", "No"])]
        if not invalid.empty:
            log_failure("Invalid Dependents values found")
            
    if "PhoneService" in df.columns:
        invalid = df[~df["PhoneService"].isin(["Yes", "No"])]
        if not invalid.empty:
            log_failure("Invalid PhoneService values found")
            
    if "Contract" in df.columns:
        invalid = df[~df["Contract"].isin(["Month-to-month", "One year", "Two year"])]
        if not invalid.empty:
            log_failure("Invalid Contract values found")
            
    if "InternetService" in df.columns:
        invalid = df[~df["InternetService"].isin(["DSL", "Fiber optic", "No"])]
        if not invalid.empty:
            log_failure("Invalid InternetService values found")
            
    if "MonthlyCharges" in df.columns:
        if (df["MonthlyCharges"] < 0).any():
            log_failure("MonthlyCharges must be non-negative")
        if (df["MonthlyCharges"] > 200).any():
            log_failure("MonthlyCharges unexpectedly high (>200)")
            
    if "tenure" in df.columns:
        if (df["tenure"] < 0).any():
            log_failure("tenure must be non-negative")
        if (df["tenure"] > 120).any():
            log_failure("tenure unexpectedly high (>120)")
            
    if "TotalCharges" in df.columns:
        # Note: TotalCharges might have spaces in the raw data, converting them to NaN first usually
        total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors='coerce')
        if (total_charges_numeric < 0).any():
            log_failure("TotalCharges must be non-negative")

    return expectations_passed, failed_expectations
    
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95  # Allow 5% exceptions for edge cases
    )
    
    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = ge_df.validate()
    
    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations
