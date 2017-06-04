Vue.component('focus-switcher', {
    template: `
    <div v-on:click="changeFocus()" v-bind:class="classObject">
      <span v-bind:uk-icon="icon"></span>
      <br/>
      <span class="uk-badge">{{nbElements}}</span>
    </div>
    `,
    props: {
        nbElements: {
            type: Number,
            default: function(){return 0}
        },
        active: {
            type: Boolean,
            default: function(){return false}
        },
        focusName: {
            type: String,
            required: true
        },
        iconName: {
            type: String,
            required: true
        },
        node: {
            required: true
        }
    },
    data: function() {
        return {
            icon: "icon: " + this.iconName,
            classObject: {
              "uk-button": true,
              "uk-button-small": true,
              "uk-button-default": true,
            },
            selected: false
        }
      },
    mounted: function(){
        if (this.active) {
          this.changeFocus()
        }
    },
    methods: {
        unSelect: function(){
            this.selected = false
            this.classObject["uk-button-default"] = true
            this.classObject["uk-button-primary"] = false
            this.classObject["vulcan-button-primary"] = false
        },
        select: function(){
            this.selected = true
            this.classObject["uk-button-default"] = false
            this.classObject["uk-button-primary"] = true
            this.classObject["vulcan-button-primary"] = true
        },
        changeFocus: function(){
            if (this.nbElements > 0) {
              this.$emit('focusChanged', this, this.node, this.focusName)
              this.select()
            }
        }
      }
});
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
                        <button class="uk-button uk-button-small uk-button-default uk-margin-remove" v-on:click="toggleNode(node)" v-if="!node.Mariana.open">
                            {{node.Mariana.data.name}}
                        </button>
                        <button class="uk-button uk-button-small uk-button-secondary vulcan-button-secondary" v-on:click="toggleNode(node)" v-else>
                            {{node.Mariana.data.name}}
                        </button>
                      <div class="uk-align-center uk-margin-remove">
                        <div class="uk-button-group">
                          <focus-switcher active v-on:focusChanged="changeFocus" focus-name="hyperParameters" icon-name="lock" v-bind:nb-elements="node.Mariana.data['hyperParameters'].size" v-bind:node="node"></focus-switcher>
                          <focus-switcher v-on:focusChanged="changeFocus" focus-name="parameters" icon-name="unlock" v-bind:nb-elements="node.Mariana.data['parameters'].size" v-bind:node="node"></focus-switcher>
                          <focus-switcher v-on:focusChanged="changeFocus" focus-name="notes" icon-name="comments" v-bind:nb-elements="node.Mariana.data['notes'].size" v-bind:node="node"></focus-switcher>
                        </div>
                      </div>
                      <ul v-show="node.Mariana.open" class="uk-list uk-list-striped">
                        <li v-for="category in node.Mariana.availableCats[node.Mariana.focus]">
                          <span class="uk-text-bold uk-text-capitalize">
                            {{category}}
                          </span>
                          <br/>
                          <span v-for="kv in node.Mariana.data[node.Mariana.focus][category]">
                            <span class="uk-text-muted">{{kv.name}}</span>:
                            <span v-if="kv.value === undefined || kv.value === null">none</span>
                            <span v-else>{{kv.value}}</span>
                            <br/>
                          </span>
                        </li>
                      </ul>
                  </li>
              </ul>
          </div>
      </div>`,

  props: ["nodesp", "edgesp", "physicsp"],
  qtemplate: `<div v-bind:style="{height: height + 'px'}" ref="network"></div>`,
  data : function(){
  createNodes = function(layers, color, highlightColor, hoverColor){
      categories = ["layer", "initializations", "activation", "regularizations", "decorators", "learningScenari"]
      superCats = ["parameters", "hyperParameters", "notes"]
      nodes = []
      namesToI = {}
      maxLvl = 0
      for (var i = 0; i < layers.length; i++) {
        availableCats = {}
        for (var k = 0; k < superCats.length; k++) { 
          availableCats[superCats[k]] = []
          for (var j = 0; j < categories.length; j++) {
            if (layers[i][1][superCats[k]][categories[j]].length > 0) {
              availableCats[superCats[k]].push(categories[j])
            };
          };
        };
        node = { id: i, label: layers[i][0], shape: "box", font: {color: "white"},
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
          level: layers[i][1].level,
          shadow:true,
          Mariana: {
              open:false,
              focus: undefined,
              focusSwitchers: new Set(),
              data: layers[i][1],
              availableCats: availableCats
            },
        }
          
        if (layers[i][1].level > maxLvl) {
          maxLvl = layers[i][1].level
        };

        namesToI[layers[i][0]] = i
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
    nodesObj = JSON.parse(this.nodesp)
    res = createNodes(nodesObj, nodeColor, nodeHighlightColor, nodeHoverColor)
    nodes = res.nodes
    nodesNamesToI = res.namesToI
    maxLvl = res.maxLvl
    nodes[0].Mariana.open = true
  
    edgesObj = JSON.parse(this.edgesp)
    edges = createEdges(edgesObj, nodesNamesToI, edgeColor)
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
        physics: this.physicsp === "true",
        drawed:false
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
    changeFocus: function(switcher, node, focus){
        node.Mariana.focus = focus;
        node.Mariana.focusSwitchers.add(switcher)
        
        switchers = Array.from(node.Mariana.focusSwitchers)
        for (var i = switchers.length - 1; i >= 0; i--) {
          if (switcher != switchers[i]) {
            switchers[i].unSelect()
          }
        }
        if (!node.Mariana.open) {
            if (this.drawed) {
              this.toggleNode(node)
            }
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
              this.nodes[i].label = this.nodes[i].Mariana.data.name + "\n" + "("+ this.nodes[i].Mariana.data.shape + ")"
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
            physics:{
              enabled: this.physics
            }
        };
        var container = this.$refs.network;
        this.network = new vis.Network(container, data, options);
        
        var self = this
        this.network.on('select', function (params) {
            for (var i = self.nodes.length - 1; i >= 0; i--) {
                self.nodes[i].Mariana.open = false
            };
            
            for (var i = params.edges.length - 1; i >= 0; i--) {
                var ed = self.edges[params.edges[i]]
                self.nodes[ed.to].Mariana.open = true
                self.nodes[ed.from].Mariana.open = true
            };

            for (var i = params.nodes.length - 1; i >= 0; i--) {
                self.nodes[params.nodes[0]].Mariana.open = true
            };
        });
        this.drawed = true
    }

  }
})

new Vue({
  el: '#all'
})