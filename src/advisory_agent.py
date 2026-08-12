import pandas as pd
import numpy as np

class AirQualityHealthAgent:
    """
    AI Agent for Air Quality Health Advisory and Risk Assessment.
    Analyzes AQI forecasts, pollutant levels, and individual health profiles
    to generate actionable, personalized medical and lifestyle precautions.
    """
    
    PROFILES = [
        "General Public",
        "Asthma / Respiratory Conditions",
        "Cardiovascular / Heart Disease",
        "Elderly (65+ years)",
        "Children & Infants",
        "Outdoor Athletes / Workers"
    ]
    
    def __init__(self):
        pass

    def get_aqi_category(self, aqi):
        if aqi <= 50: return "Good", "#00e400", "Low Risk"
        elif aqi <= 100: return "Satisfactory", "#ffff00", "Minor Risk"
        elif aqi <= 200: return "Moderate", "#ff7e00", "Moderate Risk"
        elif aqi <= 300: return "Poor", "#ff0000", "High Risk"
        elif aqi <= 400: return "Very Poor", "#99004c", "Very High Risk"
        else: return "Severe", "#7e0023", "Emergency Risk"

    def assess_health_risk(self, aqi, profile, pm25=None, pm10=None, no2=None):
        category, color_code, risk_level = self.get_aqi_category(aqi)
        
        # Calculate Profile Safety Score (100 = completely safe, 0 = extreme hazard)
        base_score = max(0, 100 - (aqi / 5.0))
        
        # Profile sensitivity multiplier
        sensitivity = {
            "General Public": 1.0,
            "Asthma / Respiratory Conditions": 1.5,
            "Cardiovascular / Heart Disease": 1.4,
            "Elderly (65+ years)": 1.3,
            "Children & Infants": 1.35,
            "Outdoor Athletes / Workers": 1.2
        }.get(profile, 1.0)
        
        custom_score = max(0, round(100 - ((100 - base_score) * sensitivity)))
        
        # Specific Pollutant Threat Analysis
        pollutant_warnings = []
        if pm25 and pm25 > 60:
            pollutant_warnings.append(f"High Fine Particulate Matter (PM2.5: {pm25:.1f} µg/m³) - Deep alveolar lung penetration hazard.")
        if pm10 and pm10 > 100:
            pollutant_warnings.append(f"Elevated Coarse Particles (PM10: {pm10:.1f} µg/m³) - Upper respiratory tract irritation.")
        if no2 and no2 > 80:
            pollutant_warnings.append(f"High Nitrogen Dioxide (NO2: {no2:.1f} µg/m³) - Increased bronchial hyper-reactivity.")
            
        # Profile-Specific Guidance
        if profile == "Asthma / Respiratory Conditions":
            if aqi <= 50:
                action = "Air quality is good. Great day for outdoor activities."
                mask = "Not required"
                purifier = "Not needed"
            elif aqi <= 100:
                action = "Acceptable air quality. Carry quick-relief inhaler as a precaution."
                mask = "Optional"
                purifier = "Keep windows ventilated"
            elif aqi <= 200:
                action = "Reduce prolonged outdoor exertion. Keep inhaler ready at all times."
                mask = "N95 / FFP2 outside"
                purifier = "Run indoor HEPA purifier on medium"
            elif aqi <= 300:
                action = "STRICT: Avoid outdoor physical activity. Stay in clean indoor environment."
                mask = "N95 mandatory outside"
                purifier = "Run HEPA purifier on High 24/7"
            else:
                action = "CRITICAL ALERT: Remain indoors with doors/windows sealed. High risk of asthma exacerbation or attack."
                mask = "N99 / Respirator for any emergency exposure"
                purifier = "HEPA Purifier + Carbon filter at Maximum speed"
                
        elif profile == "Cardiovascular / Heart Disease":
            if aqi <= 100:
                action = "Air quality is suitable for normal routine outdoor activities."
                mask = "Not required"
                purifier = "Not needed"
            elif aqi <= 200:
                action = "Avoid strenuous cardiovascular workouts outdoors. Opt for mild indoor exercise."
                mask = "Cloth/Surgical mask outdoors"
                purifier = "Recommended indoors"
            elif aqi <= 300:
                action = "High blood pressure / ischemic risk. Avoid outdoor walks or heavy lifting."
                mask = "N95 mandatory outdoors"
                purifier = "Required indoors"
            else:
                action = "CRITICAL: Strict indoor rest. Avoid any physical exertion that increases heart rate."
                mask = "N95/N99 mandatory outdoors"
                purifier = "HEPA purifier on Maximum speed"

        elif profile in ["Elderly (65+ years)", "Children & Infants"]:
            if aqi <= 100:
                action = "Safe for outdoor play and daily walks."
                mask = "Not required"
                purifier = "Not needed"
            elif aqi <= 200:
                action = "Limit strenuous outdoor play during afternoon peak pollution hours."
                mask = "Surgical mask recommended"
                purifier = "Recommended in sleeping rooms"
            else:
                action = "Keep children and elderly indoors. Seal windows during morning and evening smog."
                mask = "Well-fitting N95 mask outdoors"
                purifier = "Essential indoor HEPA filtration"

        elif profile == "Outdoor Athletes / Workers":
            if aqi <= 100:
                action = "Suitable for outdoor training and heavy manual work."
                mask = "Not required"
                purifier = "Not needed"
            elif aqi <= 200:
                action = "Shift high-intensity cardio training to early morning or indoor gym."
                mask = "Sports anti-pollution mask"
                purifier = "Recommended after workout"
            else:
                action = "Suspend intense outdoor endurance training to avoid high toxic intake."
                mask = "Respirator / N95 required for outdoor work"
                purifier = "Recommended indoors"

        else: # General Public
            if aqi <= 100:
                action = "Air quality is good to satisfactory. Enjoy normal routine."
                mask = "Not required"
                purifier = "Not needed"
            elif aqi <= 200:
                action = "Unusually sensitive people should consider reducing heavy outdoor exertion."
                mask = "Optional outdoors"
                purifier = "Recommended for bedrooms"
            elif aqi <= 300:
                action = "Avoid prolonged outdoor exposure during smog hours."
                mask = "N95 mask recommended"
                purifier = "Recommended indoors"
            else:
                action = "Everyone should avoid outdoor exertion and close windows."
                mask = "N95 mandatory outdoors"
                purifier = "HEPA purifier recommended"

        return {
            "AQI": aqi,
            "AQI_Category": category,
            "Color_Code": color_code,
            "Health_Risk_Level": risk_level,
            "Personalized_Safety_Score": custom_score,
            "Profile": profile,
            "Recommended_Action": action,
            "Mask_Guidance": mask,
            "Air_Purifier_Guidance": purifier,
            "Pollutant_Warnings": pollutant_warnings
        }

if __name__ == "__main__":
    agent = AirQualityHealthAgent()
    result = agent.assess_health_risk(245, "Asthma / Respiratory Conditions", pm25=110, no2=85)
    print("Agent Assessment Sample Output:")
    print(result)
