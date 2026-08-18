/* Small on purpose. The server renders everything; this file only
   (1) hands Vega-Lite specs to the vendored renderer, and
   (2) re-runs that after htmx swaps in a fragment carrying one. */

function renderVegaSpecs(root) {
  (root || document).querySelectorAll("script.vega-spec").forEach(function (el) {
    if (el.dataset.rendered) return;
    el.dataset.rendered = "1";
    var target = document.getElementById(el.dataset.target);
    if (!target || typeof vegaEmbed === "undefined") return;
    vegaEmbed(target, JSON.parse(el.textContent), {
      actions: false,
      renderer: "svg",
    });
  });
}

document.addEventListener("DOMContentLoaded", function () {
  renderVegaSpecs(document);
  document.body.addEventListener("htmx:afterSwap", function (evt) {
    renderVegaSpecs(evt.target.parentElement || document);
  });
});
