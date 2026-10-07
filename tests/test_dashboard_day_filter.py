"""Execute the generated dashboard JS against lightweight map/DOM doubles."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from generate_dashboard import compute_stats, generate_html, load_data


@unittest.skipUnless(shutil.which("node"), "Node.js is required for browser logic checks")
class TestDashboardDayFilter(unittest.TestCase):
    def test_day_selection_and_map_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alerts.json"
            records = {
                str(i): dict(lat="22.335", lng="114.165", address=address,
                             create_dt=timestamp, upvote="0", downvote="0")
                for i, (timestamp, address) in enumerate([
                    ("2026-10-07 00:00:00", "Today start"),
                    ("2026-10-07 23:59:59", "Today end"),
                    ("2026-10-06 23:59:59", "Yesterday"),
                    ("2026-09-30 12:00:00", "Old event"),
                    ("invalid", "Undated"),
                ])
            }
            path.write_text(json.dumps({"alerts": records, "meta": {}}))
            alerts, meta = load_data(path)
            html = generate_html(alerts, compute_stats(alerts), meta)
            script = html.split("<script>")[1].split("</script>")[0]
            harness = r'''
const assert = require('node:assert/strict');
const RealDate = Date;
global.Date = class extends RealDate {
  constructor(...args) { super(...(args.length ? args : ['2026-10-06T16:30:00Z'])); }
};
const elements = {};
global.document = {
  getElementById(id) {
    return elements[id] ||= {value: '', textContent: '', listeners: {}, attrs: {},
      appendChild() {}, addEventListener(event, fn) { this.listeners[event] = fn; },
      setAttribute(key, value) { this.attrs[key] = value; }};
  },
  createElement() { return {style: {}, addEventListener() {}}; }
};
let drawn = [];
global.L = {
  map() { return {setView() {return this;}}; },
  tileLayer() { return {addTo() {}}; },
  layerGroup() { return {addTo() {return this;}, clearLayers() {drawn = [];}}; },
  circleMarker(coords) { return {bindPopup(popup) {this.popup = popup; return this;},
    addTo() {drawn.push(this.popup); return this;}}; }
};
global.Chart = function() {};
global.setInterval = () => {};
'''
            checks = r'''
assert.equal(dateInput.value, '2026-10-07'); // Hong Kong is already on the next day.
assert.equal(drawn.length, 2);
assert(drawn.every(p => p.includes('Today')));
assert(!drawn.some(p => p.includes('Old event')));
elements.yesterdayBtn.listeners.click();
assert.equal(dateInput.value, '2026-10-06');
assert.equal(drawn.length, 1);
assert(drawn[0].includes('Yesterday'));
dateInput.value = '2026-09-30';
dateInput.listeners.change();
assert.equal(drawn.length, 1);
assert(drawn[0].includes('Old event'));
dateInput.value = '2026-10-01';
dateInput.listeners.change();
assert.equal(drawn.length, 0);
assert.match(elements.mapStatus.textContent, /No sightings/);
elements.latestDayBtn.listeners.click();
assert.equal(drawn.length, 2);
elements.allDatesBtn.listeners.click();
assert.equal(drawn.length, 5);
assert.equal(elements.allDatesBtn.attrs['aria-pressed'], 'true');
elements.todayBtn.listeners.click();
assert.equal(drawn.length, 2); // Old layers were cleared, rather than accumulated.
assert.equal(eventDay({create_dt: '2026-02-30 00:00:00'}), null);
assert.equal(eventDay({create_dt: ''}), null);
assert.equal(hongKongDay(new Date('2026-10-06T15:59:59Z')), '2026-10-06');
assert.equal(hongKongDay(new Date('2026-10-06T16:00:00Z')), '2026-10-07');
assert.equal(escapeHtml('<script>'), '&lt;script&gt;');
console.log('Dashboard day filtering passed');
'''
            js_path = Path(directory) / "check.cjs"
            js_path.write_text(harness + script + checks)
            result = subprocess.run(["node", str(js_path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_dataset_and_safe_script_embedding(self):
        html = generate_html([], compute_stats([]), {})
        self.assertIn('id="mapDate"', html)
        self.assertIn('Latest recorded day', html)
        self.assertIn('All History', html)
        alert = dict(lat=22.3, lng=114.1, address='</script><script>alert(1)</script>',
                     create_dt='', district='', region='', dt=None, hour=0, dow='Unknown')
        html = generate_html([alert], compute_stats([alert]), {})
        marker_script = html.split('const markers = ')[1].split(';')[0]
        self.assertNotIn('</script>', marker_script)
        self.assertIn('\\u003c', marker_script)


if __name__ == '__main__':
    unittest.main()
