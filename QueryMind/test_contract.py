with open("test_contract.pdf", "wb") as f:
    import fitz
    doc = fitz.open()
    
    page1 = doc.new_page()
    page1.insert_text((50, 50), """SUPPLY AGREEMENT - CONTRACT #2024-001

Parties:
- Supplier: TechParts S.r.l., Milan, Italy
- Buyer: Mediatica Group S.p.A., Rome, Italy

Contract Start Date: January 1, 2024
Expiry Date: December 31, 2025
Renewal: Automatic 12-month renewal unless terminated 60 days prior

Payment Terms: Net 30 days from invoice date
Late Payment Penalty: 2% per month
Currency: EUR

Total Contract Value: €450,000
Annual Volume: €225,000
""", fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((50, 50), """TERMS AND CONDITIONS

Clause 3 - Delivery:
All goods must be delivered within 15 business days.
Delivery location: Via Roma 123, Milan, Italy.
Shipping costs are borne by the Supplier.

Clause 4 - Warranties:
Supplier guarantees all products for 24 months.
Defective items must be replaced within 5 business days.

Clause 5 - Termination:
Either party may terminate with 60 days written notice.
Immediate termination allowed in case of material breach.

Clause 6 - Confidentiality:
All contract terms are strictly confidential for 5 years.
""", fontsize=11)

    page3 = doc.new_page()
    page3.insert_text((50, 50), """PRICING AND SLA

Product Pricing:
- Component A (SKU-001): €45.00 per unit
- Component B (SKU-002): €120.00 per unit  
- Component C (SKU-003): €890.00 per unit

Service Level Agreement:
- Uptime guarantee: 99.5%
- Response time for critical issues: 4 hours
- Scheduled maintenance: First Sunday of each month

Penalties:
- SLA breach: 5% discount on monthly invoice
- Delivery delay >5 days: €500 per day penalty

Signatures:
Supplier: Marco Rossi (CEO, TechParts S.r.l.)
Buyer: Laura Bianchi (CPO, Mediatica Group S.p.A.)
Date: January 1, 2024
""", fontsize=11)

    doc.save("test_contract.pdf")
    doc.close()
    print("PDF created: test_contract.pdf")
