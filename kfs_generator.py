from fpdf import FPDF
from datetime import date
import streamlit as st
import pandas as pd

class KFS(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Key Fact Statement (KFS)', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln(8)

def generate_kfs_pdf(applicant_name, pd_score, loan_terms, top_features):
    """
    Generates a PDF Key Fact Statement for a given applicant.
    """
    pdf = KFS()
    pdf.add_page()
    
    # Applicant & Date
    pdf.chapter_body(f"Applicant Name: {applicant_name}")
    pdf.chapter_body(f"Date of Issue: {date.today().strftime('%Y-%m-%d')}")
    pdf.ln(5)

    # Loan Details
    pdf.chapter_title('1. Loan Details')
    pdf.chapter_body(f"Loan Amount: {loan_terms['amount']}")
    pdf.chapter_body(f"Tenure: {loan_terms['tenure']} months")
    pdf.chapter_body(f"Annual Percentage Rate (APR): {loan_terms['apr']:.2f}%")
    pdf.chapter_body(f"Total Repayment Amount: {loan_terms['total_repayment']}")
    pdf.chapter_body(f"Fees and Charges: {loan_terms['fees']}")

    # Credit Assessment
    pdf.chapter_title('2. Credit Assessment Summary')
    pdf.chapter_body(f"Your Z-Score Probability of Default (PD) is: {pd_score*100:.2f}%")
    pdf.chapter_body("Based on our assessment, this is a summary of the top factors that influenced your score:")
    
    for reason, points in top_features.items():
        sign = "+" if points > 0 else ""
        pdf.chapter_body(f"  - {reason}: ({sign}{points:.1f} points)")

    # Compliance & Terms
    pdf.chapter_title('3. Important Information & Compliance')
    pdf.chapter_body("This loan is offered by a partner Regulated Entity (RE) and adheres to RBI Digital Lending Guidelines.")
    pdf.chapter_body("You are entitled to a mandatory cooling-off period of one day. You can exit the loan by repaying the principal without penalty during this period.")
    pdf.chapter_body("For any grievances, please contact our designated Grievance Officer at grievances@z-score.in.")

    pdf.output("KFS_Statement.pdf")
    return "KFS_Statement.pdf"

if __name__ == '__main__':
    # Example usage
    example_loan_terms = {
        'amount': '₹5,000',
        'tenure': 6,
        'apr': 24.0,
        'total_repayment': '₹5,600',
        'fees': '₹100'
    }
    example_features = {
        "Consistent electricity bill payments": 15,
        "Long SIM card tenure": 5,
        "Recent loan default reported by partner NGO": -40
    }
    
    filename = generate_kfs_pdf("Priya Sharma", 0.15, example_loan_terms, example_features)
    print(f"Generated sample KFS: {filename}")