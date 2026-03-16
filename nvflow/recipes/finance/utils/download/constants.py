# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Constants and configuration for SEC section extraction.

Section definitions, part mappings, and other constants used by the extractors.
"""

# 10-K sections with their official numbering
SECTIONS_10K: list[str] = [
    "1",  # Business
    "1A",  # Risk Factors
    "1B",  # Unresolved Staff Comments
    "1C",  # Cybersecurity
    "2",  # Properties
    "3",  # Legal Proceedings
    "4",  # Mine Safety Disclosures
    "5",  # Market for Registrant's Common Equity
    "6",  # Reserved (formerly Selected Financial Data)
    "7",  # Management's Discussion and Analysis
    "7A",  # Quantitative and Qualitative Disclosures About Market Risk
    "8",  # Financial Statements and Supplementary Data
    "9",  # Changes in and Disagreements With Accountants
    "9A",  # Controls and Procedures
    "9B",  # Other Information
    "9C",  # Disclosure Regarding Foreign Jurisdictions that Prevent Inspections
    "10",  # Directors, Executive Officers and Corporate Governance
    "11",  # Executive Compensation
    "12",  # Security Ownership of Certain Beneficial Owners and Management
    "13",  # Certain Relationships and Related Transactions, and Director Independence
    "14",  # Principal Accountant Fees and Services
    "15",  # Exhibits and Financial Statement Schedules
]

PARTS_10K: dict[str, list[str]] = {
    "Part I": ["1", "1A", "1B", "1C", "2", "3", "4"],
    "Part II": ["5", "6", "7", "7A", "8", "9", "9A", "9B", "9C"],
    "Part III": ["10", "11", "12", "13", "14"],
    "Part IV": ["15"],
}

# 8-K sections (event-driven, different structure)
SECTIONS_8K: list[str] = [
    "1-1",  # Entry into a Material Definitive Agreement
    "1-2",  # Termination of a Material Definitive Agreement
    "1-3",  # Bankruptcy or Receivership
    "1-4",  # Mine Safety Reporting
    "1-5",  # Material Cybersecurity Incidents
    "2-1",  # Completion of Acquisition or Disposition of Assets
    "2-2",  # Results of Operations and Financial Condition
    "2-3",  # Creation of a Direct Financial Obligation
    "2-4",  # Triggering Events That Accelerate or Increase
    "2-5",  # Costs Associated with Exit or Disposal Activities
    "2-6",  # Material Impairments
    "3-1",  # Notice of Delisting or Failure to Satisfy
    "3-2",  # Unregistered Sales of Equity Securities
    "3-3",  # Material Modification to Rights of Security Holders
    "4-1",  # Changes in Registrant's Certifying Accountant
    "4-2",  # Non-Reliance on Previously Issued Financial Statements
    "5-1",  # Changes in Control of Registrant
    "5-2",  # Departure of Directors or Certain Officers
    "5-3",  # Amendments to Articles of Incorporation or Bylaws
    "5-4",  # Temporary Suspension of Trading
    "5-5",  # Amendments to the Registrant's Code of Ethics
    "5-6",  # Change in Shell Company Status
    "5-7",  # Submission of Matters to a Vote of Security Holders
    "5-8",  # Shareholder Director Nominations
    "7-1",  # Regulation FD Disclosure
    "8-1",  # Other Events
    "9-1",  # Financial Statements and Exhibits
]

VALID_FORM_TYPES: list[str] = ["10-K", "10-Q", "8-K"]
