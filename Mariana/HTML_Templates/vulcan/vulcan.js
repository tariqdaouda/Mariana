data = {}

data.layers = {}
names = ["Input1", "Input2", "H1", "H2", "H3", "Output1", "Output2"]    

for (var i = names.length - 1; i >= 0; i--) {
	template = {
			level: undefined,
            hyperParameters: {
            	shape: "(200, 400, 10, 10)",
            },
            parameters: {
            	"Input1.W": "(500, 63)",
            	"Input1.b": "(500, )"
            },
            abstractions:{
            	GradientDescent: {
            		hyperParameters: {
	            		lr: 0.1,
	            		momentum: 0.9,
	            		reverse: false            			
            		},
            		parameters:{},
            		notes: {}
            	},
            	L1: {
            		hyperParameters: {
	            		factor: 0.1        			
            		},
            		parameters:{},
            		notes: {}
            	},
            	L2: {
            		hyperParameters: {
	            		factor: 0.1        			
            		},
            		parameters:{},
            		notes: {}
            	},
            	Batchnormalization: {
            		hyperParameters: {},
            		parameters:{
            			"Input1.Batchnormalization.W": "(500,)"
            		},
            		notes: {}
            	}
            },
            notes:{
                    note1: "Lorem ipsum dolor sit amet, consectetur adipisicing elit. Molestias, ullam? Iste, odit asperiores doloribus deleniti quos a possimus. Molestiae veritatis error ab iste itaque soluta velit laboriosam eligendi ullam consequatur?"
                },
    }
	data.layers[names[i]] = template	
}
data.layers["Input1"].level = 0
data.layers["Input2"].level = 0
data.layers["H1"].level = 1
data.layers["H2"].level = 1
data.layers["H3"].level = 2
data.layers["Output1"].level = 3
data.layers["Output2"].level = 3

data.edges = [
	{from: "Input1", to: "H1"},
	{from: "Input2", to: "H2"},
	{from: "H1", to: "H3"},
	{from: "H2", to: "H3"},
	{from: "H3", to: "Output1"},
	{from: "H3", to: "Output2"},
]


toKV = function(obj) {
	res = []
	keys = Object.keys(obj)
	for (var i = keys.length - 1; i >= 0; i--) {
		res.push({name: keys[i], value:obj[keys[i]]})
	}
	return res
}

Vue.component('graph-view', {
	template: `
	    <div uk-grid>
	        <div class="uk-card uk-card-default uk-width-3-4@l uk-width-3-4@m uk-width-1-1@s">
	            <div class="uk-button-group">
	                <button class="uk-button uk-button-medium uk-button-primary vulcan-button-primary" v-on:click="changeDirection('UD')">
	                    <span uk-icon="icon: arrow-down; ratio: 2"></span>
	                </button>
	                <button class="uk-button uk-button-medium uk-button-primary vulcan-button-primary" v-on:click="changeDirection('DU')">
	                    <span uk-icon="icon: arrow-up; ratio: 2"></span>
	                </button>
	                <button class="uk-button uk-button-medium uk-button-primary vulcan-button-primary" v-on:click="changeDirection('LR')">
	                    <span uk-icon="icon: arrow-right; ratio: 2"></span>
	                </button>
	                <button class="uk-button uk-button-medium uk-button-primary vulcan-button-primary" v-on:click="changeDirection('RL')">
	                    <span uk-icon="icon: arrow-left; ratio: 2"></span>
	                </button>
	            </div>
	            <button class="uk-button uk-button-medium uk-button-primary vulcan-button-primary uk-align-right" v-on:click="toggleShapes()">
	                <span uk-icon="icon: code; ratio: 2"></span>
	            </button>
	            <div v-bind:style="{height: height + 'px'}" ref="network"></div>
	            <div class="uk-button-group">
	                <button class="uk-button uk-button-medium uk-button-primary vulcan-button-primary" v-on:click="decreaseHeight()">
	                    <span uk-icon="icon: triangle-up; ratio: 2"></span>
	                </button>
	                <button class="uk-button uk-button-medium uk-button-primary vulcan-button-primary" v-on:click="increaseHeight()">
	                    <span uk-icon="icon: triangle-down; ratio: 2"></span>
	                </button>
	            </div>
	        </div>

	        <div class="uk-width-1-4@l uk-width-1-4@m uk-width-1-1@s">
	            <ul uk-accordion="multiple: true">
	                <li v-for="node in nodes">
                        <button class="uk-button uk-button-small uk-button-default" v-on:click="toggleNode(node)" v-if="!node.Mariana.open">
                            <span uk-icon="icon: chevron-down; ratio: 0.75"></span>
                            {{node.Mariana.data.name}}
                        </button>
                        <button class="uk-button uk-button-small uk-button-secondary vulcan-button-secondary" v-on:click="toggleNode(node)" v-else>
                            <span uk-icon="icon: chevron-down; ratio: 0.75"></span>
                            {{node.Mariana.data.name}}
                        </button>
	                    <div uk-dropdown="pos: bottom-right">
	                        <ul class="uk-nav uk-dropdown-nav">
	                            <ul class="uk-list uk-list-striped">
	                                <li v-on:click="changeFocus(node, 'hyperParameters')"><a>Hyper-parameters</a></li>
	                                <li v-on:click="changeFocus(node, 'parameters')"><a>Parameters</a></li>
	                                <li v-on:click="changeFocus(node, 'notes')"><a>Notes</a></li>
	                            </ul>
	                        </ul>
	                    </div>
	                    <ul v-show="node.Mariana.open" class="uk-list uk-list-striped">
	                        <li v-for="thing in node.Mariana.data[node.Mariana.focus]">
	                        	<span class="uk-text-bold">{{thing.name}}</span> <br/> <span v-for="kv in thing.value"> {{kv.name}}: {{kv.value}}<br/></span>
	                        </li>
	                    </ul>
	                </li>
	            </ul>
	        </div>

	    </div>`,
  props: ["nodes", "edges", "physics"],
  qtemplate: `<div v-bind:style="{height: height + 'px'}" ref="network"></div>`,
  data : function(){
	createNodes = function(layers, color, highlightColor, hoverColor){
  		nodes = []
  		namesToI = {}
  		maxLvl = 0
      layerKV = toKV(layers)
  		for (var i = 0; i < layerKV.length; i++) {
  			layer = layerKV[i]
	    	mData = { name: layer.name, shape: layer.value.shape, parameters: [], hyperParameters: [], notes: [] }
  			
  			mData.parameters.push({name: "self", value: toKV(layer.value.parameters)})
  			mData.hyperParameters.push({name: "self", value: toKV(layer.value.hyperParameters)})
  			mData.notes.push({name: "self", value: toKV(layer.value.notes)})

  			abs = toKV(layer.value.abstractions)
	  		for (var j = abs.length - 1; j >= 0; j--) {
	  			if (abs[j].value.parameters) {
	  				v = toKV(abs[j].value.parameters)
		  			if (v.length > 0) {
	  					mData.parameters.push({name: abs[j].name, value: v})
		  			}
	  			}
	  			if (abs[j].value.hyperParameters) {
	  				mData.hyperParameters.push({name: abs[j].name, value: toKV(abs[j].value.hyperParameters)})
	  			}
	  			if (abs[j].value.notes) {
	  				v = toKV(toKV(abs[j].value.notes))
	  				if (v.length > 0) {
		  				mData.notes.push({name: abs[j].name, value: v})
					}
	  			}
	  		}
	  		node = { id: i, label: layer.name, shape: "box", font: {color: "white"},
	  			color: {
	  				background: color,
	  				border: "black",
	  				highlight:{
	  					background:highlightColor,
	  					border:"black"
	  				},
	  				hover:{
	  					background:hoverColor,
	  					border:"black"
	  				}
	  			},
	  			level: layer.value.level,
	  			shadow:true,
	  			Mariana: {open:false, focus: "hyperParameters", data: mData}
	  		}
	        
	        if (layer.value.level > maxLvl) {
	            maxLvl = layer.value.level
	        };
	        namesToI[layer.name] = i
	        nodes.push(node)
  		}
      return {nodes: nodes, maxLvl: maxLvl, namesToI}
  	}

  	createEdges = function(edges, namesToI, color) {
  		for (var i = edges.length - 1; i >= 0; i--) {
  			edges[i].to = namesToI[edges[i].to]
  			edges[i].from = namesToI[edges[i].from]
  			edges[i].id = i
  			edges[i].shadow = true
  			edges[i].arrows = 'to'
  			edges[i].color = color
  		}
  		return edges
  	}

  	nodeColor = 'rgba(25, 131, 165, 0.5)'
  	nodeHighlightColor = 'rgba(25, 131, 165, 1)'
  	nodeHoverColor = 'rgba(25, 131, 165, 1)'
  
  	edgeColor = "#f68920"
    // res = createNodes(data.layers, nodeColor, nodeHighlightColor, nodeHoverColor)
  	res = createNodes(this.nodes, nodeColor, nodeHighlightColor, nodeHoverColor)
    nodes = res.nodes
    nodesNamesToI = res.namesToI
  	maxLvl = res.maxLvl
    nodes[0].Mariana.open = true
	
  // edges = createEdges(data.edges, nodesNamesToI, edgeColor)
	edges = createEdges(this.edges, nodesNamesToI, edgeColor)
    ret = {
        nodes: nodes,
        nodeColor: nodeColor,
        nodesNamesToI: nodesNamesToI,
        edges: edges,
        edgeColor: edgeColor,
        network: null,
        maxLvl: maxLvl,
        height: (maxLvl +1) * 150,
        direction: "UD",
        showShapes: true,
        physics: true
    }
    return ret
  },
  mounted: function(){
    this.draw()
  },
  methods: {
    increaseHeight: function(){
        this.height = this.height + 150
        this.draw()
    },
    decreaseHeight: function(){
        this.height = this.height - 150
        this.draw()
    },
    toggleNode: function(node){
        if (!node.Mariana.open) {
            node.Mariana.open=true
            this.network.selectNodes([node.id])
        }else{
            node.Mariana.open=false
        }
    },
    changeFocus: function(node, focus){
        node.Mariana.focus = focus;
        if (!node.Mariana.open) {
            this.toggleNode(node)
        }
    },
    toggleShapes: function(){
        this.showShapes = !this.showShapes
        this.draw()
    },
    changeDirection: function(dir){
        this.direction = dir
        this.draw()
    },
    destroy: function(){
        if (this.network !== null) {
            this.network.destroy();
        }
        this.network = null;
    },
    draw: function() {
        this.destroy();
        for (var i = this.nodes.length - 1; i >= 0; i--) {
        	if (this.showShapes) {
                this.nodes[i].label = this.nodes[i].Mariana.data.name + "\n" + this.nodes[i].Mariana.data.hyperParameters[0].value[0].value + this.nodes[i].level + " " +this.nodes[i].id
            }else{
                this.nodes[i].label = this.nodes[i].Mariana.data.name
            };
        };
        
        var data = {
            nodes: this.nodes,
            edges: this.edges
        };

        var options = {
            edges: {
                smooth: {
                    type: 'cubicBezier',
                    forceDirection: (this.direction == "UD" || this.direction == "DU") ? 'vertical' : 'horizontal',
                    roundness: 0.4
                }
            },
            layout: {
                hierarchical: {
                    direction: this.direction
                }
            },
            physics:this.physics
        };
        var container = this.$refs.network;
        this.network = new vis.Network(container, data, options);
        console.log(this.nodesNamesToI)
        var self = this
        this.network.on('select', function (params) {
            for (var i = self.nodes.length - 1; i >= 0; i--) {
                self.nodes[i].Mariana.open = false
            };
            
            for (var i = params.edges.length - 1; i >= 0; i--) {
                var ed = self.edges[params.edges[i]]
                // console.log(self.nodesNamesToI[ed.from] + "->"+ self.nodesNamesToI[ed.to])
                self.nodes[ed.to].Mariana.open = true
                self.nodes[ed.from].Mariana.open = true
            };

            for (var i = params.nodes.length - 1; i >= 0; i--) {
                self.nodes[params.nodes[0]].Mariana.open = true
            };

        });
    }

  }
})

new Vue({
  el: '#all'
})