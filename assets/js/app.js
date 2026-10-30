// Portfolio Lab browser script: equations, the settings drawer, number inputs,
// keyboard shortcuts, the work card and lazy tables. No framework, no network
// service: every request goes to this computer's server.
"use strict";

(() => {
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const drawerRoot = document.getElementById("pl-drawer-root");
  const same = { credentials: "same-origin" };

  // ---------------------------------------------------------------- equations

  /** Typeset display equations and inline `$...$` mathematics with the bundled KaTeX. */
  function renderMath(root) {
    if (!window.katex) return;
    for (const element of root.querySelectorAll(".pl-latex:not([data-rendered])")) {
      element.dataset.rendered = "1";
      katex.render(element.textContent, element, { displayMode: true, throwOnError: false });
    }
    if (!window.renderMathInElement) return;
    for (const element of root.querySelectorAll(".pl-inline-math:not([data-rendered])")) {
      element.dataset.rendered = "1";
      renderMathInElement(element, { delimiters: [{ left: "$", right: "$", display: false }], throwOnError: false });
    }
  }

  // ---------------------------------------------------------------- number inputs

  /** Show a number in the Python format of its field: ".2f" or "g". */
  function formatNumber(value, format) {
    const fixed = /^\.(\d+)f$/.exec(format || "");
    if (fixed) return value.toFixed(Number(fixed[1]));
    if (format === "g") return String(Number(value.toPrecision(6)));
    return String(value);
  }

  function setNumber(input, value) {
    input.value = formatNumber(value, input.dataset.format);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  /** The minus and plus buttons and the arrow keys add one step, within the limits. */
  function stepNumber(input, direction) {
    const step = Number(input.dataset.step) || 1;
    const low = input.min === "" ? -Infinity : Number(input.min);
    const high = input.max === "" ? Infinity : Number(input.max);
    const current = Number.parseFloat(input.value);
    const start = Number.isFinite(current) ? current : Math.max(low, 0);
    const next = Math.min(high, Math.max(low, Number((start + direction * step).toFixed(10))));
    setNumber(input, next);
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest(".pl-step");
    if (!button) return;
    const input = button.parentElement.querySelector("input");
    if (input.disabled) return;
    stepNumber(input, Number(button.dataset.stepBy));
    input.focus();
  });

  document.addEventListener("keydown", (event) => {
    const input = event.target;
    if (!(input instanceof HTMLInputElement) || input.type !== "number" || !input.dataset.step) return;
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      stepNumber(input, event.key === "ArrowUp" ? 1 : -1);
    }
  });

  document.addEventListener(
    "blur",
    (event) => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement) || input.type !== "number" || !input.dataset.format) return;
      const value = Number.parseFloat(input.value);
      if (Number.isFinite(value) && input.checkValidity()) input.value = formatNumber(value, input.dataset.format);
    },
    true,
  );

  // ---------------------------------------------------------------- landing limits

  const preview = document.querySelector("form[data-preview]");
  let previewTimer = 0;

  /** Save the landing page's limits on the server, as the Streamlit widgets do on every change. */
  function sendPreview() {
    clearTimeout(previewTimer);
    previewTimer = 0;
    if (!preview) return Promise.resolve();
    const body = new URLSearchParams({ target_vol: preview.elements.target_vol.value, cap: preview.elements.cap.value });
    return fetch("/api/preview", { method: "POST", body, ...same }).catch(() => undefined);
  }

  if (preview) {
    preview.addEventListener("input", () => {
      clearTimeout(previewTimer);
      previewTimer = setTimeout(sendPreview, 300);
    });
    preview.addEventListener("change", sendPreview);
  }

  // ---------------------------------------------------------------- settings drawer

  let openedAt = 0;
  let returnFocus = null;
  let opening = false;

  const drawerOpen = () => drawerRoot.childElementCount > 0;

  /** Open "Change the study" with controls from the applied setup. */
  async function openDrawer() {
    if (drawerOpen() || opening || document.body.dataset.page === "work") return;
    opening = true;
    try {
      returnFocus = document.activeElement;
      await sendPreview();
      const response = await fetch("/drawer", same);
      if (!response.ok) return;
      drawerRoot.innerHTML = await response.text();
      openedAt = performance.now();
      document.body.classList.add("pl-drawer-open");
      const form = drawerRoot.querySelector("#setup_form");
      setupDrawer(form);
      localizeKeys(drawerRoot);
      renderMath(drawerRoot);
      drawerRoot.querySelector(".pl-drawer").focus();
    } finally {
      opening = false;
    }
  }

  function closeDrawer() {
    drawerRoot.innerHTML = "";
    document.body.classList.remove("pl-drawer-open");
    if (returnFocus instanceof HTMLElement) returnFocus.focus({ preventScroll: true });
  }

  /** Esc or a click outside. The second click of a double click on Settings lands
   *  outside the new drawer, so a dismiss in the first 0.5 s is ignored. */
  function dismissDrawer() {
    if (performance.now() - openedAt >= 500) closeDrawer();
  }

  function setupDrawer(form) {
    const differs = form.querySelector("#pl-differs");
    let draftTimer = 0;
    let draftRequest = 0;

    /** Show the controls of the selected market; hidden controls are disabled and keep their stored values. */
    function sync() {
      const market = form.querySelector('input[name="market"]:checked')?.value ?? "";
      const values = {
        market,
        history: form.querySelector('input[name="history"]:checked')?.value ?? "",
        frequency: form.querySelector('select[name="frequency"]')?.value ?? "",
      };
      for (const [control, current] of Object.entries(values)) {
        for (const block of form.querySelectorAll(`[data-when-${control}]`)) {
          block.hidden = !block.getAttribute(`data-when-${control}`).split("|").includes(current);
        }
      }
      for (const control of form.querySelectorAll("input, select, textarea")) {
        control.disabled = control.closest("[hidden]") !== null || (control.id === "compare_market" && market === "Demo");
      }
    }

    async function checkDraft() {
      const request = ++draftRequest;
      try {
        const response = await fetch(form.dataset.draft, { method: "POST", body: new FormData(form), ...same });
        const result = await response.json();
        if (request === draftRequest) differs.hidden = !result.differs;
      } catch {
        // The notice is optional; Build portfolio still applies the draft.
      }
    }

    function scheduleDraft() {
      clearTimeout(draftTimer);
      draftTimer = setTimeout(checkDraft, 200);
    }

    function showFile(input) {
      const field = input.closest(".pl-file");
      const name = input.files && input.files[0] ? input.files[0].name : "";
      field.querySelector("[data-file-name]").textContent = name || "Drag and drop file here";
      const stored = field.querySelector("[data-stored]");
      if (stored) stored.hidden = Boolean(name) || stored.dataset.removed === "1";
    }

    form.addEventListener("change", (event) => {
      const target = event.target;
      if (target.name === "market" && target.value === "Demo") {
        // change_market: the six-asset demo needs a higher holding limit.
        const cap = form.querySelector("#cap");
        cap.value = formatNumber(20, cap.dataset.format);
      }
      if (target.dataset.mirror) {
        form.querySelector(`input[type="hidden"][name="${target.dataset.mirror}"]`).value = String(target.checked);
      }
      if (target.type === "file") showFile(target);
      sync();
      scheduleDraft();
    });
    form.addEventListener("input", scheduleDraft);
    form.addEventListener("click", (event) => {
      const remove = event.target.closest("[data-remove]");
      if (!remove) return;
      form.querySelector(`input[name="${remove.dataset.remove}"]`).value = "1";
      const stored = remove.closest("[data-stored]");
      stored.dataset.removed = "1";
      stored.hidden = true;
      scheduleDraft();
    });
    sync();
  }

  drawerRoot.addEventListener("click", (event) => {
    if (event.target.closest("[data-close]")) closeDrawer();
    else if (event.target.closest("[data-dismiss]")) dismissDrawer();
  });

  /** Keep Tab inside the open drawer. */
  drawerRoot.addEventListener("keydown", (event) => {
    if (event.key !== "Tab") return;
    const focusable = [...drawerRoot.querySelectorAll("a[href], button, input, select, textarea, summary, [tabindex]")].filter(
      (element) => !element.disabled && element.tabIndex >= 0 && element.offsetParent !== null,
    );
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });

  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-open-setup]")) {
      event.preventDefault();
      openDrawer();
    }
  });

  // ---------------------------------------------------------------- navigation and shortcuts

  function currentView() {
    return { overview: "Portfolio", method: "Data & method" }[document.body.dataset.page] ?? "";
  }

  function scrollToTarget(anchor) {
    if (anchor === "top") {
      window.scrollTo({ top: 0, behavior: "instant" });
      return;
    }
    const target = document.getElementById(anchor);
    if (!target) return;
    target.scrollIntoView({ behavior: reduceMotion ? "instant" : "smooth", block: "start" });
    target.querySelector("h2")?.focus({ preventScroll: true });
  }

  /** `jump(view, anchor)`: scroll in place, or switch the view on the server and scroll after the load. */
  function jump(view, anchor) {
    if (currentView() === view) {
      scrollToTarget(anchor);
      return;
    }
    const form = document.createElement("form");
    form.method = "post";
    form.action = "/view";
    for (const [name, value] of Object.entries({ view, anchor: anchor === "top" ? "" : anchor })) {
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = name;
      input.value = value;
      form.append(input);
    }
    document.body.append(form);
    form.submit();
  }

  const KEYS = { 1: ["Portfolio", "top"], 2: ["Portfolio", "pl-market"], 4: ["Data & method", "top"] };
  const NOT_TYPING = new Set(["checkbox", "radio", "range", "button", "file", "reset", "submit", "color"]);

  /** Single-key shortcuts do not fire while the user types, as in Streamlit. */
  function typing(target) {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable || target.tagName === "TEXTAREA" || target.tagName === "SELECT") return true;
    return target.tagName === "INPUT" && !NOT_TYPING.has(target.type);
  }

  document.addEventListener("keydown", (event) => {
    if (event.defaultPrevented) return;
    if (event.key === "Escape" && drawerOpen()) {
      event.preventDefault();
      dismissDrawer();
      return;
    }
    const modifier = isMac ? event.metaKey : event.ctrlKey;
    if (event.key === "Enter" && modifier && !event.altKey && !event.shiftKey) {
      const button = (drawerOpen() ? drawerRoot : document.querySelector("main"))?.querySelector("[data-build]");
      if (button && !button.disabled) {
        event.preventDefault();
        if (button.form) button.form.requestSubmit(button);
        else button.click();
      }
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey || typing(event.target) || drawerOpen()) return;
    const key = event.key.toLowerCase();
    if (key === "s" && document.querySelector("[data-open-setup]")) {
      event.preventDefault();
      openDrawer();
    } else if (key === "e" && document.querySelector("[data-export]")) {
      event.preventDefault();
      document.querySelector("[data-export]").click();
    } else if (KEYS[key] && document.body.dataset.ready === "1") {
      event.preventDefault();
      jump(...KEYS[key]);
    }
  });

  /** Export: download the zip, or show the server's message when the export fails. */
  async function exportReport(link) {
    if (link.dataset.busy) return;
    link.dataset.busy = "1";
    try {
      const response = await fetch(link.href, same);
      if (!response.ok) {
        window.location.assign(link.href);
        return;
      }
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = link.getAttribute("download") || "portfolio-lab.zip";
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch {
      window.location.assign(link.href);
    } finally {
      delete link.dataset.busy;
    }
  }

  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-export]");
    if (!link) return;
    event.preventDefault();
    exportReport(link);
  });

  // The active view segment needs no request.
  for (const button of document.querySelectorAll('.pl-view-switch button[aria-pressed="true"]')) {
    button.addEventListener("click", (event) => event.preventDefault());
  }

  // ---------------------------------------------------------------- work card

  const hero = document.querySelector("[data-poll]");

  /** Follow the build. When it ends, load the result or the error page. */
  async function poll() {
    try {
      const response = await fetch(hero.dataset.poll, { ...same, cache: "no-store" });
      const state = await response.json();
      if (!state.active) {
        window.location.replace("/");
        return;
      }
      hero.querySelector("#pl-step p").textContent = state.step;
      const progress = hero.querySelector("#work_progress");
      progress.hidden = !state.bar;
      if (state.bar) {
        progress.querySelector(".pl-progress-text").textContent = state.bar.text;
        const track = progress.querySelector(".pl-progress-track");
        track.setAttribute("aria-valuenow", String(Math.round(state.bar.value * 100)));
        track.querySelector("i").style.width = `${(state.bar.value * 100).toFixed(1)}%`;
      }
      const detail = hero.querySelector("#pl-detail");
      detail.hidden = !state.detail;
      if (state.detail) detail.lastElementChild.textContent = state.detail;
    } catch {
      // Keep polling: the server may be busy for a moment.
    }
    setTimeout(poll, 400);
  }

  // ---------------------------------------------------------------- lazy tables

  document.addEventListener(
    "toggle",
    async (event) => {
      const details = event.target;
      if (!(details instanceof HTMLDetailsElement) || !details.open || !details.dataset.src || details.dataset.loaded) return;
      details.dataset.loaded = "1";
      const slot = details.querySelector("[data-lazy]");
      try {
        const response = await fetch(details.dataset.src, same);
        slot.innerHTML = await response.text();
      } catch {
        slot.textContent = "The table could not be loaded. Reload the page and try again.";
      }
    },
    true,
  );

  // ---------------------------------------------------------------- start

  /** Show the Command key on a Mac, as Streamlit's "Mod" shortcuts do. */
  function localizeKeys(root) {
    if (!isMac) return;
    for (const key of root.querySelectorAll("[data-mod-key]")) key.textContent = key.textContent.replace("Ctrl", "⌘");
  }

  function start() {
    localizeKeys(document);
    renderMath(document);
    // Typeset equations change the layout, so return to the target of a view switch.
    const target = window.location.hash && document.getElementById(window.location.hash.slice(1));
    if (target) target.scrollIntoView({ block: "start" });
    if (hero) setTimeout(poll, 300);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
