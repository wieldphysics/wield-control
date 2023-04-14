function cyto_run(){
    fetch('graph')
        .then(response => response.json())
        .then(graph_json => {
            console.log(graph_json);

            const json_str = JSON.stringify(graph_json, null, 2); // take json from cytoscape
            document.getElementById("json").innerText = json_str;

            var cy = (window.cy = cytoscape({
                container: document.getElementById("cy"),
                style: [
                    {
                        selector: 'node',
                        css: {
                            'content': 'data(id)',
                            'text-valign': 'top',
                            'text-halign': 'center'
                        }
                    },
                    {
                        selector: ':parent',
                        css: {
                            'text-valign': 'top',
                            'text-halign': 'center',
                        }
                    },
                    {
                        selector: 'edge',
                        css: {
                            'curve-style': 'bezier',
                            'target-arrow-shape': 'triangle'
                        }
                    }
                ],

                elements: graph_json,

                layout: {
                    name: 'preset',
                    padding: 5
                }
            }));

            cy.ready(function() {
                cy.on('dragfree', 'node', function(evt){
                    var node = evt.target;

                    const json = cy.json(); // take json from cytoscape
                    const json_str = JSON.stringify(json, null, 2); // take json from cytoscape
                    // download json as you want
                    const data = "text/json;charset=utf-8," + encodeURIComponent(json_str);
                    const a = document.getElementById('link');
                    a.href = 'data:' + data;
                    a.download = 'data.json';
                    a.innerHTML = 'download JSON';

                    document.getElementById("json").innerText = json_str;

                    fetch('/graph_recv', {
                        credentials: "same-origin",
                        mode: "same-origin",
                        method: "post",
                        headers: { "Content-Type": "application/json" },
                        body: json_str
                    });
                });
            });
        });
}

if (document.readyState === 'loading') {  // Loading hasn't finished yet
    document.addEventListener('DOMContentLoaded', cyto_run);
} else {  // `DOMContentLoaded` has already fired
    cyto_run();
}


var saveAsSvg = function(filename) {
		var svgContent = cy.svg({scale: 1, full: true});
		var blob = new Blob([svgContent], {type:"image/svg+xml;charset=utf-8"});
		saveAs(blob, "demo.svg");
};
var getSvgUrl = function() {
		var svgContent = cy.svg({scale: 1, full: true});
		var blob = new Blob([svgContent], {type:"image/svg+xml;charset=utf-8"});
		var url = URL.createObjectURL(blob);
		return url;
};
