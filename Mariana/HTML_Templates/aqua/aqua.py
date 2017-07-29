TEMPLATE = r"""
    <script>
      window.onload = function() {
        var g = graphlibDot.parse(
          #DOT#
        )

        var renderer = new dagreD3.Renderer();
        renderer.run(g, d3.select("svg g"));


        var svg = document.querySelector('#graphContainer');
        var bbox = svg.getBBox();
        svg.style.width = bbox.width + 40.0 + "px";
        svg.style.height = bbox.height + 40.0 + "px";
      }
    </script>
    <style>
      .footer {
        color: #0074D9;
      }

      .header{
        color: #0074D9;
      }

      svg {
        overflow: hidden;
      }

      .node rect {
        stroke: #0074D9;
        stroke-width: 2px;
        fill: #7FDBFF;
      }

      .edgeLabel rect {
        fill: #7FDBFF;
      }

      .edgePath {
        stroke: #0074D9;
        stroke-width: 2px;
        fill: none;
      }
    </style>
<html>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.4.11/d3.min.js"></script>
  <script src="http://cpettitt.github.io/project/graphlib-dot/v0.4.10/graphlib-dot.min.js"></script>
  <script src="http://cpettitt.github.io/project/dagre-d3/v0.1.5/dagre-d3.min.js"></script>
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css" integrity="sha512-dTfge/zgoMYpP7QbHy4gWMEGsbsdZeCXz7irItjcC3sPUFtf0kuFbDz/ixG7ArTxmDjLXDmezHubeNikyKGVyQ==" crossorigin="anonymous">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">

  <body>
    <div class = "container text-center">
      <h1 class="header"> #NAME# <small>#DATE#</small> </h1>
      <div class="jumbotron">
        <svg id="graphContainer"> <g/> </svg>
      </div>
      <a href="http://www.github.com/tariqdaouda/Mariana">
        <i class="fa fa-github-alt"></i>
        <h5 class="footer"> <i class="fa fa-hand-spock-o"></i> Mariana generated <i class="fa fa-hand-spock-o"></i> </h5>
      </a>
    </div>
  </body>
</html>"""

def getHTML(dotString, name, date):
  dot = dotString.split("\n")[1:]
  for l in dot :
    l.replace("\n", "\\n");
  dot[0] = "'" + dot[0]
  dot[-1] = dot[-1] + "'"
  dotF = "' +\n'".join(dot).replace("_>", "->")

  temp = TEMPLATE.replace("#DOT#", dotF).replace("#NAME#", name).replace("#DATE#", date)
  return temp