const express = require("express");
const puppeteer = require("puppeteer");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const OpenAI = require("openai");              // ← same SDK, pointed at xAI

const app = express();
app.use(cors());
app.use(express.json());

// ── Grok client ──────────────────────────────────────────────────────────────
// Install: npm install openai
// Get key: https://console.x.ai/
const grok = new OpenAI({
  apiKey: "YOUR_API_KEY",             // ← paste your xAI key here
  baseURL: "https://api.x.ai/v1",
});
const GROK_TEXT_MODEL = "grok-3-latest";       // text-only model for form filling

const DUMMY_SITE_PATH = 'C:\\Users\\Dell\\OneDrive\\Desktop\\govt-miniproject\\pan_card_assistant\\dummy-site';
const DUMMY_SITE_URL = "file:///" + DUMMY_SITE_PATH.replace(/\\/g, "/") + "/index.html";


// ════════════════════════════════════════
// PAYMENT STATUS STORE
// ════════════════════════════════════════
let paymentStatus = { status: "pending", ref: "", applicant: "" };

app.get("/payment-status", (req, res) => {
  res.json(paymentStatus);
});

app.post("/payment-reset", (req, res) => {
  paymentStatus = { status: "pending", ref: "", applicant: "" };
  res.json({ ok: true });
});


// ════════════════════════════════════════
// ASK GROK  (replaces askGemini)
// ════════════════════════════════════════
async function askGrok(page, userdata, currentStep, instruction) {
  const stepHTML = await page.$eval(`#step${currentStep}`, el => el.innerHTML);
  console.log(`🤖 Grok analyzing Step ${currentStep}...`);

  const prompt = `You are an AI agent filling a government PAN card application form.
This is Step ${currentStep} of 6.

EXACT HTML of current step:
<form_html>${stepHTML}</form_html>

User data:
<user_data>${JSON.stringify(userdata, null, 2)}</user_data>

Field mapping:
STEP 1: first_name→#firstName, middle_name→#middleName, last_name→#lastName, first+last→#nameOnCard, gender→#gender(Male/Female/Other), dob→#dob(YYYY-MM-DD use setdate action), aadhaar_number→#aadhaar, residentState→#residentState
STEP 2: identity proof→#poiDoc(Aadhaar), address proof→#poaDoc(Aadhaar), dob proof→#dobDoc(Aadhaar)
STEP 3: mobile→#mobile, email→#email, fatherFirstName→#fatherFirstName, fatherLastName→#fatherLastName
STEP 4: flat_no→#resFlatNo, building→#resBuilding, road→#resRoad, locality→#resLocality, city→#resCity, district→#resDistrict, residentState→#resState, pincode→#resPincode, income checkboxes→use check action: incomeSalary/incomeCapitalGains/incomeHouseProperty/incomeOtherSources/incomeBusiness/incomeNone
STEP 5: areaCode→#areaCode, aoType→#aoType, rangeCode→#rangeCode, aoNumber→#aoNumber, rep_first→#repFirstName, rep_last→#repLastName, rep_area→#repLocality, rep_district→#repCity, rep_state→#repState, pincode→#repPincode, declaration→#declaration(Himself/Herself), verifier→#verifierName(use first+last if empty), verification_place→#verificationPlace(use city if empty)
Next button: #step${currentStep} .btn-primary

Task: ${instruction}

RULES:
1. Only fill fields with non-empty values
2. Always end with click on #step${currentStep} .btn-primary
3. Use exact option VALUE from HTML for selects
4. If declaration empty always use "Himself/Herself"
5. If verifier empty use first_name + last_name
6. If verification_place empty use city
7. For DOB field use action "setdate" not "fill"
8. For checkboxes use action "check"
9. Return ONLY valid JSON array, no markdown, no explanation

Format: [{"action":"fill","selector":"#firstName","value":"Prakash"},{"action":"setdate","selector":"#dob","value":"1990-05-15"},{"action":"check","selector":"#incomeNone"},{"action":"click","selector":"#step${currentStep} .btn-primary"}]`;

  try {
    const response = await grok.chat.completions.create({
      model: GROK_TEXT_MODEL,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 1500,
      temperature: 0,
    });

    const raw     = response.choices[0].message.content.trim();
    const cleaned = raw.replace(/```json|```/g, "").trim();
    const actions = JSON.parse(cleaned);
    if (!actions.some(a => a.selector && a.selector.includes("btn-primary")))
      actions.push({ action: "click", selector: `#step${currentStep} .btn-primary` });
    return actions;

  } catch (err) {
    console.warn(`⚠️ Grok error step ${currentStep}: ${err.message} — using fallback`);
    return getFallbackActions(currentStep, userdata);
  }
}


// ════════════════════════════════════════
// VERIFY & FIX (uses Grok)
// ════════════════════════════════════════
async function verifyAndFix(page, data, step) {
  console.log(`🔍 Verifying Step ${step}...`);

  const filled = await page.evaluate((stepId) => {
    const el = document.getElementById(stepId);
    if (!el) return {};
    const result = {};
    el.querySelectorAll("input, select, textarea").forEach(el => {
      if (el.id) result[el.id] = el.value;
    });
    return result;
  }, `step${step}`);

  const missing = Object.entries(filled).filter(([,v]) => !v || v.trim() === "").map(([k]) => k);
  if (missing.length === 0) { console.log("✅ Verification passed"); return; }

  console.log(`⚠️ Empty fields: ${missing.join(", ")} — asking Grok to fix...`);

  const fixPrompt = `Fix these empty form fields: ${missing.join(", ")}
User data: ${JSON.stringify(data, null, 2)}
Field mapping: #firstName→first_name, #lastName→last_name, #nameOnCard→first+last, #gender→gender, #dob→dob(use setdate action), #aadhaar→aadhaar_number, #mobile→mobile, #email→email, #resFlatNo→flat_no, #resBuilding→building, #resRoad→road, #resLocality→locality, #resCity→city, #resPincode→pincode, #fatherFirstName→fatherFirstName or father_first_name, #fatherLastName→fatherLastName or father_last_name, #repFirstName→rep_first, #repLastName→rep_last, #verifierName→use first+last, #verificationPlace→use city, #declaration→always "Himself/Herself"
For DOB use action "setdate". Return ONLY JSON array of fix actions. No Next button. Return [] if nothing to fix.`;

  try {
    const response = await grok.chat.completions.create({
      model: GROK_TEXT_MODEL,
      messages: [{ role: "user", content: fixPrompt }],
      max_tokens: 800,
      temperature: 0,
    });

    const raw   = response.choices[0].message.content.trim();
    const fixes = JSON.parse(raw.replace(/```json|```/g, "").trim());
    if (fixes.length > 0) {
      console.log(`🔧 Fixing ${fixes.length} field(s)...`);
      await executeActions(page, fixes);
    }
  } catch (err) {
    console.warn("⚠️ Verify/fix error:", err.message);
  }
}


// ════════════════════════════════════════
// FALLBACK ACTIONS (unchanged)
// ════════════════════════════════════════
function getFallbackActions(step, data) {
  const fatherFirst = data.fatherFirstName || data.father_first_name || "";
  const fatherLast  = data.fatherLastName  || data.father_last_name  || "";
  const fullName    = `${data.first_name || ""} ${data.last_name || ""}`.trim();

  const incomeActions = [];
  if (data.income_salary)         incomeActions.push({ action: "check", selector: "#incomeSalary" });
  if (data.income_capital_gains)  incomeActions.push({ action: "check", selector: "#incomeCapitalGains" });
  if (data.income_house_property) incomeActions.push({ action: "check", selector: "#incomeHouseProperty" });
  if (data.income_other_sources)  incomeActions.push({ action: "check", selector: "#incomeOtherSources" });
  if (data.income_business)       incomeActions.push({ action: "check", selector: "#incomeBusiness" });
  if (data.income_no_income)      incomeActions.push({ action: "check", selector: "#incomeNone" });
  if (incomeActions.length === 0) incomeActions.push({ action: "check", selector: "#incomeNone" });

  const fallbacks = {
    1: [
      { action: "fill",    selector: "#firstName",     value: data.first_name     || "" },
      { action: "fill",    selector: "#middleName",    value: data.middle_name    || "" },
      { action: "fill",    selector: "#lastName",      value: data.last_name      || "" },
      { action: "fill",    selector: "#nameOnCard",    value: fullName },
      { action: "select",  selector: "#gender",        value: data.gender         || "Male" },
      { action: "setdate", selector: "#dob",           value: data.dob            || "" },
      { action: "fill",    selector: "#aadhaar",       value: data.aadhaar_number || "" },
      { action: "select",  selector: "#residentState", value: data.residentState  || "Kerala" },
      { action: "click",   selector: "#step1 .btn-primary" }
    ],
    2: [
      { action: "select", selector: "#poiDoc", value: "Aadhaar" },
      { action: "select", selector: "#poaDoc", value: "Aadhaar" },
      { action: "select", selector: "#dobDoc", value: "Aadhaar" },
      { action: "click",  selector: "#step2 .btn-primary" }
    ],
    3: [
      { action: "fill",  selector: "#mobile",          value: data.mobile || "" },
      { action: "fill",  selector: "#email",           value: data.email  || "" },
      { action: "fill",  selector: "#fatherFirstName", value: fatherFirst },
      { action: "fill",  selector: "#fatherLastName",  value: fatherLast },
      { action: "click", selector: "#step3 .btn-primary" }
    ],
    4: [
      { action: "fill",   selector: "#resFlatNo",   value: data.flat_no       || "" },
      { action: "fill",   selector: "#resBuilding", value: data.building      || "" },
      { action: "fill",   selector: "#resRoad",     value: data.road          || "" },
      { action: "fill",   selector: "#resLocality", value: data.locality      || "" },
      { action: "fill",   selector: "#resCity",     value: data.city          || "" },
      { action: "fill",   selector: "#resDistrict", value: data.district      || "" },
      { action: "select", selector: "#resState",    value: data.residentState || "Kerala" },
      { action: "fill",   selector: "#resPincode",  value: data.pincode       || "" },
      ...incomeActions,
      { action: "click",  selector: "#step4 .btn-primary" }
    ],
    5: [
      { action: "fill",   selector: "#areaCode",          value: data.areaCode           || "" },
      { action: "fill",   selector: "#aoType",            value: data.aoType             || "" },
      { action: "fill",   selector: "#rangeCode",         value: data.rangeCode          || "" },
      { action: "fill",   selector: "#aoNumber",          value: data.aoNumber           || "" },
      { action: "fill",   selector: "#repFirstName",      value: data.rep_first          || "" },
      { action: "fill",   selector: "#repLastName",       value: data.rep_last           || "" },
      { action: "fill",   selector: "#repLocality",       value: data.rep_area           || "" },
      { action: "fill",   selector: "#repCity",           value: data.rep_district       || "" },
      { action: "fill",   selector: "#repState",          value: data.rep_state          || "" },
      { action: "fill",   selector: "#repPincode",        value: data.pincode            || "" },
      { action: "select", selector: "#declaration",       value: "Himself/Herself" },
      { action: "fill",   selector: "#verifierName",      value: data.verifier  || fullName },
      { action: "fill",   selector: "#verificationPlace", value: data.verification_place || data.city || "" },
      { action: "click",  selector: "#step5 .btn-primary" }
    ],
    6: [{ action: "click", selector: "#step6 .btn-primary" }]
  };

  return fallbacks[step] || [{ action: "click", selector: `#step${step} .btn-primary` }];
}


// ════════════════════════════════════════
// EXECUTE ACTIONS ON PAGE (unchanged)
// ════════════════════════════════════════
async function executeActions(page, actions) {
  for (const action of actions) {
    try {
      const exists = await page.$(action.selector);
      if (!exists) { console.warn(`  ⚠️ Not found: ${action.selector}`); continue; }

      if (action.action === "fill") {
        await page.$eval(action.selector, el => el.value = "");
        await page.type(action.selector, String(action.value || ""));
        console.log(`  ✅ Filled  ${action.selector} = "${action.value}"`);

      } else if (action.action === "setdate") {
        await page.evaluate((sel, val) => {
          const el = document.querySelector(sel);
          if (el) {
            el.value = val;
            el.dispatchEvent(new Event('input',  { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
          }
        }, action.selector, String(action.value || ""));
        console.log(`  ✅ SetDate ${action.selector} = "${action.value}"`);

      } else if (action.action === "select") {
        await page.select(action.selector, String(action.value || ""));
        console.log(`  ✅ Select  ${action.selector} = "${action.value}"`);

      } else if (action.action === "click") {
        await page.click(action.selector);
        console.log(`  ✅ Clicked ${action.selector}`);
        await new Promise(r => setTimeout(r, 800));

      } else if (action.action === "check") {
        await page.evaluate((sel) => {
          const el = document.querySelector(sel);
          if (el) {
            el.checked = true;
            el.dispatchEvent(new Event('change', { bubbles: true }));
          }
        }, action.selector);
        console.log(`  ✅ Checked ${action.selector}`);

      } else if (action.action === "upload") {
        const el = await page.$(action.selector);
        if (el && action.value && fs.existsSync(action.value))
          await el.uploadFile(action.value);
      }

      await new Promise(r => setTimeout(r, 300));
    } catch (err) {
      console.warn(`  ⚠️ Failed  ${action.selector}: ${err.message}`);
    }
  }
}


// ════════════════════════════════════════
// AUTOFILL ROUTE — Agentic AI
// ════════════════════════════════════════
app.post("/autofill", async (req, res) => {
  const data = req.body;
  console.log("📥 Received confirmed user data");

  paymentStatus = { status: "pending", ref: "", applicant: "" };

  let browser;
  try {
    if (!fs.existsSync(DUMMY_SITE_PATH))
      throw new Error("Dummy site not found: " + DUMMY_SITE_PATH);

    browser = await puppeteer.launch({
      headless: false,
      defaultViewport: null,
      args: ["--start-maximized"]
    });

    const page = await browser.newPage();
    console.log("🌐 Opening dummy PAN site...");
    await page.goto(DUMMY_SITE_URL, { waitUntil: "domcontentloaded", timeout: 15000 });
    await new Promise(r => setTimeout(r, 1500));

    const title = await page.title();
    console.log("📄 Page title:", title);

    const step1Exists = await page.$("#step1");
    if (!step1Exists) throw new Error("Page loaded but #step1 not found.");
    console.log("✅ Dummy site loaded\n");

    // Close popup if present
    const popup = await page.$("#referencePopup");
    if (popup) {
      await page.click("#referencePopup button");
      await new Promise(r => setTimeout(r, 500));
    }

    // STEP 1
    const s1 = await askGrok(page, data, 1, "Fill personal details: name, gender, DOB (use setdate action), Aadhaar, resident state. Then click Next.");
    await executeActions(page, s1);
    await verifyAndFix(page, data, 1);
    await page.click("#step1 .btn-primary").catch(() => {});
    await new Promise(r => setTimeout(r, 800));
    console.log("✅ Step 1 done\n");

    // STEP 2
    const s2 = await askGrok(page, data, 2, "Select Aadhaar for Identity, Address and DOB proof. Then click Next.");
    await executeActions(page, s2);
    await new Promise(r => setTimeout(r, 800));
    console.log("✅ Step 2 done\n");

    // STEP 3
    const s3 = await askGrok(page, data, 3, "Fill mobile, email and father name (use fatherFirstName and fatherLastName from data). Then click Next.");
    await executeActions(page, s3);
    await verifyAndFix(page, data, 3);
    await page.click("#step3 .btn-primary").catch(() => {});
    await new Promise(r => setTimeout(r, 800));
    console.log("✅ Step 3 done\n");

    // STEP 4
    const s4 = await askGrok(page, data, 4, "Fill all address fields: flat, building, road, locality, city, district, state, pincode. Also check the correct income checkboxes. Then click Next.");
    await executeActions(page, s4);
    await verifyAndFix(page, data, 4);
    await page.click("#step4 .btn-primary").catch(() => {});
    await new Promise(r => setTimeout(r, 800));
    console.log("✅ Step 4 done\n");

    // STEP 5
    const s5 = await askGrok(page, data, 5, "Fill AO code, representative assessee, declaration, verifier name and place. Then click Next.");
    await executeActions(page, s5);
    await verifyAndFix(page, data, 5);
    await page.click("#step5 .btn-primary").catch(() => {});
    await new Promise(r => setTimeout(r, 800));
    console.log("✅ Step 5 done\n");

    // STEP 6 — Upload files
    console.log("📎 Uploading documents...");
    const resolvePath = (p) => {
      if (!p) return null;
      const variants = [p, p.replace(/\//g, "\\"), p.replace(/\\/g, "/")];
      for (const v of variants) { if (fs.existsSync(v)) return v; }
      console.warn("  ❌ File not found:", p);
      return null;
    };

    const uploadAndShow = async (selector, statusId, filePath) => {
      const resolvedPath = resolvePath(filePath);
      const el = await page.$(selector);
      if (el && resolvedPath) {
        await el.uploadFile(resolvedPath);
        const fname = resolvedPath.split(/[\\/]/).pop();
        await page.evaluate((sid, fn) => {
          const s = document.getElementById(sid);
          if (s) { s.innerText = "✅ " + fn + " — uploaded"; s.style.color = "#2e7d32"; }
        }, statusId, fname);
        console.log(`  ✅ ${selector} = ${fname}`);
      } else {
        console.warn(`  ❌ Skipped ${selector} — file not found`);
      }
    };

    await uploadAndShow("#filePOI",       "statusPOI",   data.aadhaar_path);
    await uploadAndShow("#filePOA",       "statusPOA",   data.aadhaar_path);
    await uploadAndShow("#fileDOB",       "statusDOB",   data.aadhaar_path);
    await uploadAndShow("#filePhoto",     "statusPhoto", data.photo_path);
    await uploadAndShow("#fileSignature", "statusSig",   data.signature_path);
    await new Promise(r => setTimeout(r, 800));
    await page.click("#step6 .btn-primary");
    await new Promise(r => setTimeout(r, 800));
    console.log("✅ Step 6 done\n");

    console.log("🎉 Reached payment page — watching for payment completion...");
    res.json({ status: "success", message: "AI agent filled all steps. User is now on payment page." });

    // Watch for payment success
    try {
      await page.waitForFunction(
        () => {
          const el = document.getElementById("paymentSuccess");
          return el && el.style.display !== "none" && el.style.display !== "";
        },
        { timeout: 600000 }
      );

      const ref = await page.evaluate(() => {
        const el = document.getElementById("paymentRef");
        return el ? el.innerText.trim() : "";
      });

      const applicant = `${data.first_name || ""} ${data.last_name || ""}`.trim();
      console.log(`💳 Payment complete! Ref: ${ref}, Applicant: ${applicant}`);
      paymentStatus = { status: "paid", ref, applicant };

      await new Promise(r => setTimeout(r, 3000));
      await browser.close();
      console.log("🔒 Browser closed after payment");

    } catch (watchErr) {
      console.log("⏰ Payment watch timed out or browser closed by user");
      paymentStatus = { status: "timeout", ref: "", applicant: "" };
    }

  } catch (err) {
    console.error("❌ Agent error:", err.message);
    if (!res.headersSent)
      res.status(500).json({ status: "error", message: err.message });
    if (browser) await browser.close();
  }
});


// ════════════════════════════════════════
// START SERVER
// ════════════════════════════════════════
app.listen(4000, () => {
  console.log("🚀 Agentic AI service running on port 4000");
  console.log(
    fs.existsSync(DUMMY_SITE_PATH)
      ? "✅ Dummy site found: " + DUMMY_SITE_PATH
      : "❌ Dummy site NOT found: " + DUMMY_SITE_PATH
  );
});