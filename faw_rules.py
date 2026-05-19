# faw_rules.py
# Rule-based insights engine for Fall Armyworm (Spodoptera frugiperda) management
# Based on FAW Management Guidelines (RCPC Region VI / WVSU-CICT)
# Outputs Markdown-formatted strings (bold, italic) for frontend rendering.

def get_rule_based_insights(detected_stages, overall_risk):
    """
    Generate analysis and treatment recommendations based on detected FAW life stages.
    Returns Markdown strings for 'analysis' and 'treatment'.
    """
    stages = set(detected_stages)

    # ------------------------------------------------------------------
    # Build analysis text (Markdown)
    # ------------------------------------------------------------------
    analysis_parts = []

    if not stages:
        analysis_parts.append("No FAW life stages were confidently detected in this image.")
    else:
        analysis_parts.append(f"**Detected FAW stages:** {', '.join(stages)}. **Overall risk level:** {overall_risk}.")

        stage_details = {
            "egg": ("*Eggs* do not feed but are laid in large clusters (**950–1030 eggs/female** on maize) with **94–95% hatchability** within 2–3 days. This allows rapid population buildup."),
            "larva": ("*Larvae* are the **only damaging stage**. They feed on leaves, whorls, tassels, ears, and silks for **14–22 days** (up to 30 days). **Yield losses of 10–25%** are documented. This stage presents the **highest economic risk**."),
            "pupa": ("*Pupae* reside in soil (1–3 cm depth) for **9–15 days**. They do not feed but enable transition to reproductive adults. No diapause, so continuous breeding is possible."),
            "moth": ("*Adult moths* are nocturnal, live **7–13 days**, and lay **~950 eggs** each. They drive dispersal and multiple generations (life cycle ~30 days). **Eradication is not feasible**.")
        }
        for stage in stages:
            if stage in stage_details:
                analysis_parts.append(stage_details[stage])

        if "larva" in stages:
            analysis_parts.append("**The larval stage requires immediate action to prevent crop damage.**")
        elif "egg" in stages and "larva" not in stages:
            analysis_parts.append("*Eggs are present but no larvae yet* – this is the **best window for preventive biological control**.")
        elif "pupa" in stages and "larva" not in stages:
            analysis_parts.append("*Pupae indicate the next generation of moths will emerge soon.* Soil interventions can reduce future infestations.")
        elif "moth" in stages and "larva" not in stages:
            analysis_parts.append("*Moths indicate active reproduction and spread.* Monitoring and mass trapping are recommended.")

    analysis = " ".join(analysis_parts)

    # ------------------------------------------------------------------
    # Build treatment recommendations (Markdown)
    # ------------------------------------------------------------------
    treatment_sections = []

    # General IPM principle
    treatment_sections.append(
        "### **Integrated Pest Management (IPM)**\n"
        "Combine **preventive, cultural, biological, and chemical strategies**. "
        "Prioritize **management over eradication**. Follow BPI **PAMS** framework: "
        "*Prevention, Avoidance, Monitoring, Suppression*."
    )

    # Stage-specific actionable recommendations
    if "egg" in stages:
        treatment_sections.append(
            "#### **Egg Stage (Moderate risk)**\n"
            "- **Release egg parasitoids** weekly starting at germination:\n"
            "  - *Telenomus remus*: 4,000 adults/acre\n"
            "  - *Trichogramma pretiosum*: 16,000 adults/acre\n"
            "  - *Trichogramma chilonis*: 50,000 adults/acre at 7 & 14 days after planting\n"
            "- Apply **neem‑based formulations** (Azadirachtin 1500 ppm or 5% NSKE at 5 ml/L) as *oviposition deterrent*.\n"
            "- **Do NOT apply chemical insecticides** at this stage – conserve natural enemies."
        )

    if "larva" in stages:
        treatment_sections.append(
            "#### **Larval Stage (High risk) – CRITICAL INTERVENTION WINDOW**\n"
            "- **Economic threshold:** 2.0 larvae per 10 plants (0.2/plant). If exceeded, intervene immediately.\n"
            "- **Biological control** (preferred for early instars 1st–3rd):\n"
            "  - *Bacillus thuringiensis* (Bt) – Protecto 9.4% WP at 200 g/100 L, apply to whorl at early whorl stage.\n"
            "  - *Metarhizium anisopliae* – 5 g/L (1×10⁸ cfu/g) or *Beauveria bassiana* – 250 g/100 L (Biossiana 2.5% WP).\n"
            "  - *Steinernema carpocapsae* or *Heterorhabditis indica* – 2000–3000 IJs/ml to whorl (most effective at 40 days after sowing).\n"
            "  - *Neem* (5 ml/L) – antifeedant & growth regulator.\n"
            "- **Chemical control** (only if threshold exceeded):\n"
            "  - *Emamectin benzoate* (Speedo 5.7% WG) – 25, 35, 45 days after sowing.\n"
            "  - *Chlorantraniliprole* (0.4 ml/L) – seedling & whorl stages.\n"
            "  - *Spinosad* (0.3–0.5 ml/L) or *Spinetoram* (0.5 ml/L) – early whorl at 10–20% infestation.\n"
            "- **ROTATE** insecticides with different IRAC groups to prevent resistance.\n"
            "- *Handpicking & destruction* of visible larvae (especially for late instars) is effective on small farms."
        )

    if "pupa" in stages:
        treatment_sections.append(
            "#### **Pupal Stage (Low–Moderate risk) – BETWEEN‑CROP INTERVENTION**\n"
            "- **Deep ploughing** after harvest to expose pupae to sunlight, desiccation & predators.\n"
            "- In zero‑tillage systems: incorporate **neem cake** at 500 kg/ha.\n"
            "- Soil application of *Metarhizium anisopliae* or *Beauveria bassiana*, or entomopathogenic nematodes (*Heterorhabditis indica*)."
        )

    if "moth" in stages:
        treatment_sections.append(
            "#### **Adult Stage (Moderate–High dispersal risk)**\n"
            "- **Pheromone traps** for monitoring & mass trapping:\n"
            "  - *Detection*: 4–5 traps/ha, replace lures every 6 weeks.\n"
            "  - *Mass trapping*: 15 traps/acre to suppress male population.\n"
            "- Install traps at **1–1.5 m height** inside or at field edge.\n"
            "- Record weekly captures – increasing trends indicate need for larval scouting.\n"
            "- Foliar chemical (if needed): *Thiamethoxam + Lambda‑cyhalothrin* (0.25 ml/L) – but focus on larval stage."
        )

    # Cultural / preventive measures (always include)
    treatment_sections.append(
        "### **Cultural & Preventive Measures** (apply at farm level)\n"
        "- **Synchronous planting** at community level – avoid staggered planting that creates continuous host availability.\n"
        "- **Crop diversification**: Intercrop maize with pigeon pea, black gram, or green gram. Plant *Napier grass* as border trap crop.\n"
        "- **Field sanitation**: Remove weeds (alternative hosts) and volunteer plants.\n"
        "- **Deep ploughing** after each harvest to destroy pupae.\n"
        "- Use **balanced fertilizer** and maintain optimal plant health – healthy crops tolerate damage better.\n"
        "- Erect **bird perches** (10 per acre) to attract natural predators."
    )

    # Monitoring reminder
    treatment_sections.append(
        "### **Monitoring Protocol** (BPI‑recommended)\n"
        "- **Weekly field scouting** using **W‑pattern** with 5 inspection points per field.\n"
        "- At each stop, inspect at least 10 plants for egg masses, larvae, and damage symptoms (window‑pane feeding, whorl damage, frass).\n"
        "- Record **GPS coordinates** of pheromone traps and infestation hotspots."
    )

    # Safety and resistance management
    treatment_sections.append(
        "### **Safety & Resistance Management**\n"
        "- Enter field only **48 hours after pesticide application**.\n"
        "- Observe **pre‑harvest interval of 30 days** after last chemical spray.\n"
        "- **Rotate insecticides** with different IRAC groups – do not use same mode of action repeatedly.\n"
        "- **Avoid mixing** *Spinetoram* or *Thiodicarb* with entomopathogenic fungi (*Metarhizium*/*Beauveria*) – causes >50% inhibition.\n"
        "- *Emamectin benzoate* and *Chlorantraniliprole* are compatible with *Metarhizium anisopliae*.\n"
        "- When using egg parasitoids, alternate with neem sprays at weekly intervals – **do NOT apply simultaneously**."
    )

    if not stages:
        treatment_sections = [
            "**No specific FAW life stage detected.** General preventive measures are recommended:\n"
            "- Install **pheromone traps** (4–5 per hectare) for early warning.\n"
            "- Scout fields **weekly** using the **W‑pattern** method (5 stops, ≥10 plants/stop).\n"
            "- Practice **synchronous planting**, crop rotation, and field sanitation.\n"
            "- If FAW is suspected, submit a **clearer image** of the pest or damage symptoms (e.g., larvae with inverted Y‑mark, fresh whorl damage).\n"
            "- Contact your **Regional Crop Protection Center (RCPC)** for on‑site validation."
        ]

    treatment = "\n\n".join(treatment_sections)

    # Additional urgent note if larvae present
    if "larva" in stages and overall_risk == "High":
        analysis += " **Immediate intervention is required to prevent economic yield loss (10–25% or more).**"
        treatment += "\n\n> **⚠️ URGENT:** Larvae detected at high risk level. If infestation exceeds 2 larvae per 10 plants, apply an approved chemical (Emamectin benzoate or Chlorantraniliprole) following label rates, but first consider biological options if larvae are early instars. Combine with handpicking for larger larvae."

    return {
        "analysis": analysis,
        "treatment": treatment
    }