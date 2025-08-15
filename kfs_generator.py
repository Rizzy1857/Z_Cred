from fpdf import FPDF
from datetime import date, timedelta
import os

class KFS(FPDF):
    """
    Key Fact Statement PDF generator according to Phase 2 specification.
    Implements compliance requirements and RBI Digital Lending Guidelines.
    """
    
    def header(self):
        """PDF header with title and compliance information."""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 12, 'Key Fact Statement (KFS)', 0, 1, 'C')
        self.set_font('Arial', 'I', 8)
        self.cell(0, 6, 'Generated in compliance with RBI Digital Lending Guidelines', 0, 1, 'C')
        self.ln(8)

    def footer(self):
        """PDF footer with page numbers and compliance notice."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        """Formatted section title."""
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, title, 0, 1, 'L', True)
        self.ln(2)

    def chapter_body(self, body):
        """Formatted body text."""
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln(3)

    def add_section_break(self):
        """Visual section break."""
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(4)

def generate_kfs_pdf(applicant_name, risk_scores, loan_terms, top_features, trust_scores=None, output_dir='data'):
    """
    Generates a comprehensive KFS PDF according to Phase 2 specification.
    
    Args:
        applicant_name: Applicant's full name
        risk_scores: Dict with PD scores from both models
        loan_terms: Dict with loan offer details
        top_features: Dict with top contributing factors and their SHAP values
        trust_scores: Optional dict with trust bar components
        output_dir: Directory to save PDF
    
    Returns:
        Path to generated PDF file
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize PDF
    pdf = KFS()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # === HEADER INFORMATION ===
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, f"Applicant: {applicant_name}", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Date of Issue: {date.today().strftime('%d %B %Y')}", 0, 1)
    pdf.cell(0, 6, f"KFS Reference: KFS-{date.today().strftime('%Y%m%d')}-{hash(applicant_name) % 10000:04d}", 0, 1)
    
    pdf.add_section_break()
    
    # === 1. LOAN DETAILS ===
    pdf.chapter_title('1. Loan Offer Details')
    
    pdf.chapter_body(f"Loan Amount: {loan_terms.get('amount', 'Not specified')}")
    pdf.chapter_body(f"Loan Tenure: {loan_terms.get('tenure', 'Not specified')} months")
    pdf.chapter_body(f"Annual Percentage Rate (APR): {loan_terms.get('apr', 0):.2f}%")
    pdf.chapter_body(f"Total Repayment Amount: {loan_terms.get('total_repayment', 'Not calculated')}")
    pdf.chapter_body(f"Processing Fees: {loan_terms.get('fees', 'As applicable')}")
    
    # Monthly installment calculation
    if loan_terms.get('amount') and loan_terms.get('tenure'):
        try:
            principal = float(loan_terms['amount'].replace('₹', '').replace(',', ''))
            tenure = int(loan_terms['tenure'])
            apr = float(loan_terms.get('apr', 24)) / 100
            monthly_rate = apr / 12
            
            if monthly_rate > 0:
                emi = principal * monthly_rate * (1 + monthly_rate)**tenure / ((1 + monthly_rate)**tenure - 1)
                pdf.chapter_body(f"Estimated Monthly Installment (EMI): ₹{emi:,.0f}")
        except:
            pass
    
    pdf.add_section_break()
    
    # === 2. CREDIT ASSESSMENT SUMMARY ===
    pdf.chapter_title('2. Credit Risk Assessment')
    
    # Risk scores
    lr_score = risk_scores.get('score_logistic', 0)
    xgb_score = risk_scores.get('score_xgb', 0)
    avg_score = (lr_score + xgb_score) / 2
    
    pdf.chapter_body(f"Your Z-Score Probability of Default (PD): {avg_score*100:.2f}%")
    
    # Risk categorization
    if avg_score < 0.3:
        risk_category = "Low Risk"
        risk_description = "Your profile indicates strong creditworthiness based on alternative data assessment."
    elif avg_score < 0.7:
        risk_category = "Medium Risk"  
        risk_description = "Your profile shows moderate risk factors that have been considered in the loan terms."
    else:
        risk_category = "High Risk"
        risk_description = "Your profile indicates higher risk factors. Consider improving key metrics before reapplying."
    
    pdf.chapter_body(f"Risk Category: {risk_category}")
    pdf.chapter_body(f"Assessment: {risk_description}")
    
    pdf.ln(3)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, "Key Factors Influencing Your Score:", 0, 1)
    pdf.set_font('Arial', '', 9)
    
    # Top contributing factors
    for factor, impact in list(top_features.items())[:5]:
        impact_direction = "positively" if impact > 0 else "negatively"
        impact_strength = "strongly" if abs(impact) > 0.1 else "moderately"
        
        # Convert technical names to user-friendly descriptions
        factor_descriptions = {
            'on_time_payment_ratio': 'Payment punctuality',
            'avg_bill_delay_days': 'Bill payment timing',
            'community_endorsements': 'Community standing',
            'stable_location_ratio': 'Location stability',
            'sim_card_tenure_months': 'Mobile service tenure',
            'prev_loans_defaulted': 'Previous loan history',
            'prev_loans_taken': 'Lending experience',
            'recharge_frequency_per_month': 'Mobile usage pattern'
        }
        
        friendly_name = factor_descriptions.get(factor, factor.replace('_', ' ').title())
        pdf.chapter_body(f"• {friendly_name}: {impact_strength} impacts your score {impact_direction}")
    
    pdf.add_section_break()
    
    # === 3. TRUST SCORE BREAKDOWN (if available) ===
    if trust_scores:
        pdf.chapter_title('3. Trust Score Components')
        
        behavioral = trust_scores.get('behavioral_trust', 0)
        social = trust_scores.get('social_trust', 0)
        digital = trust_scores.get('digital_trace', 0)
        total = trust_scores.get('total_trust_score', 0)
        
        pdf.chapter_body(f"Total Trust Score: {total:.1f}/100")
        pdf.chapter_body(f"• Behavioral Trust: {behavioral:.1f}/100 (Payment behavior and financial discipline)")
        pdf.chapter_body(f"• Social Trust: {social:.1f}/100 (Community endorsements and social standing)")
        pdf.chapter_body(f"• Digital Trust: {digital:.1f}/100 (Digital footprint and stability indicators)")
        
        # Graduation threshold information
        pdf.ln(2)
        if total >= 70:
            pdf.chapter_body("🎉 Congratulations! You have achieved our Trust Score graduation threshold of 70%. This may lead to better loan terms in future applications.")
        else:
            needed = 70 - total
            pdf.chapter_body(f"ℹ️  You are {needed:.1f} points away from our Trust Score graduation threshold of 70%. Focus on improving payment punctuality and community engagement.")
        
        pdf.add_section_break()
    
    # === 4. REGULATORY COMPLIANCE & IMPORTANT INFORMATION ===
    pdf.chapter_title('4. Important Information & Regulatory Compliance')
    
    pdf.chapter_body("Lending Partner: This loan is offered by our Regulated Entity (RE) partner in full compliance with RBI Digital Lending Guidelines 2022.")
    
    pdf.chapter_body("Data Usage: Your assessment is based on alternative data sources including mobile usage patterns, bill payment history, and community endorsements. All data processing has been conducted with your explicit consent.")
    
    pdf.chapter_body("Cooling-off Period: You are entitled to a mandatory cooling-off period of ONE (1) DAY from the date of this KFS. You may exit the loan application by declining the offer within this period without any penalty or charges.")
    
    cooling_off_end = date.today() + timedelta(days=1)
    pdf.chapter_body(f"Cooling-off Period Ends: {cooling_off_end.strftime('%d %B %Y')} at 11:59 PM")
    
    pdf.chapter_body("Interest Rate: The APR mentioned above is the effective rate including all charges. No additional hidden charges will be levied.")
    
    pdf.chapter_body("Grievance Redressal: For any grievances or complaints, please contact our Grievance Officer:")
    pdf.chapter_body("📧 Email: grievances@z-score.in")
    pdf.chapter_body("📞 Phone: +91-11-4567-8900 (10 AM to 6 PM, Monday to Friday)")
    pdf.chapter_body("📍 Address: Z-Score Financial Services, 123 Financial District, New Delhi - 110001")
    
    pdf.add_section_break()
    
    # === 5. TERMS & CONDITIONS HIGHLIGHTS ===
    pdf.chapter_title('5. Key Terms & Conditions')
    
    pdf.chapter_body("• Loan approval is subject to final verification and documentation.")
    pdf.chapter_body("• Early repayment is allowed without any prepayment penalty.")
    pdf.chapter_body("• Late payment charges: 2% per month on overdue amount.")
    pdf.chapter_body("• Loan disbursement will be made directly to your verified bank account.")
    pdf.chapter_body("• You have the right to receive loan statements and maintain transaction records.")
    pdf.chapter_body("• This assessment is valid for 30 days from the date of issue.")
    
    pdf.add_section_break()
    
    # === 6. NEXT STEPS ===
    pdf.chapter_title('6. Next Steps')
    
    if avg_score < 0.7:  # Approved or under review
        pdf.chapter_body("1. Review this KFS carefully during the cooling-off period.")
        pdf.chapter_body("2. If you wish to proceed, respond with your acceptance within 30 days.")
        pdf.chapter_body("3. Complete documentation and KYC verification process.")
        pdf.chapter_body("4. Loan will be disbursed upon successful verification.")
    else:  # High risk
        pdf.chapter_body("1. Your application requires additional review.")
        pdf.chapter_body("2. Consider improving key metrics mentioned in Section 2.")
        pdf.chapter_body("3. You may reapply after 90 days with updated information.")
        pdf.chapter_body("4. Contact our support team for guidance on improving your profile.")
    
    # === FOOTER COMPLIANCE ===
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.chapter_body("This document is auto-generated by Z-Score Credit Assessment Platform in compliance with RBI Digital Lending Guidelines. All information is based on data provided by the applicant and verified through alternative data sources.")
    
    pdf.chapter_body(f"Generated on: {date.today().strftime('%d %B %Y')} | Version: 2.0 | Classification: Confidential")
    
    # Save PDF
    filename = f"KFS_{applicant_name.replace(' ', '_')}_{date.today().strftime('%Y%m%d')}.pdf"
    output_path = os.path.join(output_dir, filename)
    
    try:
        pdf.output(output_path)
        print(f"📄 KFS generated successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error generating KFS: {e}")
        return None

def generate_kfs_for_applicant(applicant_data, prediction_results, trust_scores=None):
    """
    Convenience function to generate KFS from applicant data and model results.
    
    Args:
        applicant_data: Dict with applicant information
        prediction_results: Results from model_pipeline.predict_default_probability()
        trust_scores: Optional trust score breakdown
    
    Returns:
        Path to generated KFS PDF
    """
    
    # Extract applicant name
    applicant_name = applicant_data.get('name', 'Unknown Applicant')
    
    # Prepare risk scores
    risk_scores = {
        'score_logistic': prediction_results.get('score_logistic', 0),
        'score_xgb': prediction_results.get('score_xgb', 0)
    }
    
    # Determine loan terms based on risk level
    avg_score = prediction_results.get('average_score', 0)
    
    if avg_score < 0.3:  # Low risk
        loan_terms = {
            'amount': '₹15,000',
            'tenure': 12,
            'apr': 22.0,
            'total_repayment': '₹16,800',
            'fees': '₹150'
        }
    elif avg_score < 0.7:  # Medium risk
        loan_terms = {
            'amount': '₹10,000',
            'tenure': 6,
            'apr': 26.0,
            'total_repayment': '₹11,300',
            'fees': '₹200'
        }
    else:  # High risk
        loan_terms = {
            'amount': '₹5,000',
            'tenure': 3,
            'apr': 30.0,
            'total_repayment': '₹5,750',
            'fees': '₹250'
        }
    
    # Prepare top features (mock SHAP values if not available)
    top_features = prediction_results.get('top_features', {
        'Payment History': 0.15,
        'Community Standing': 0.08,
        'Location Stability': 0.05,
        'Mobile Usage Pattern': -0.03,
        'Previous Defaults': -0.12
    })
    
    # Generate KFS
    return generate_kfs_pdf(
        applicant_name=applicant_name,
        risk_scores=risk_scores,
        loan_terms=loan_terms,
        top_features=top_features,
        trust_scores=trust_scores
    )

if __name__ == '__main__':
    # Example usage with Phase 2 specification data
    print("🧪 Testing KFS Generator...")
    
    # Sample applicant data
    sample_applicant = {
        'name': 'Priya Sharma',
        'avg_bill_delay_days': 3,
        'on_time_payment_ratio': 0.85,
        'prev_loans_taken': 1,
        'prev_loans_defaulted': 0,
        'community_endorsements': 4,
        'sim_card_tenure_months': 36,
        'recharge_frequency_per_month': 15,
        'stable_location_ratio': 0.9
    }
    
    # Sample prediction results
    sample_prediction = {
        'score_logistic': 0.18,
        'score_xgb': 0.15,
        'average_score': 0.165
    }
    
    # Sample trust scores
    sample_trust = {
        'behavioral_trust': 82,
        'social_trust': 80,
        'digital_trace': 88,
        'total_trust_score': 83.3
    }
    
    # Generate sample KFS
    kfs_path = generate_kfs_for_applicant(sample_applicant, sample_prediction, sample_trust)
    
    if kfs_path:
        print(f"✅ Sample KFS generated: {kfs_path}")
    else:
        print("❌ Failed to generate sample KFS")
        
    print("\n🎉 KFS Generator test complete!")