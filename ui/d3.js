var radius = 32,
    arena_width = 2400,
    arena_height = 2400,
    edge_width = 48,
    scale = 0.2,
    c_dim = 0.8;

var x = d3.scale.linear()
    .domain([0, arena_width])
    .range([0, arena_height]);

var chart = d3.select(".chart")
    .attr("width", arena_width)
    .attr("height", arena_height);

// Background
chart.append("rect")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("fill", d3.rgb(255*.15,255*.15,255*.15));

// Border region
var edge_color = d3.rgb(128*c_dim, 128*c_dim, 128*c_dim);
var border_group = chart.append("g");
border_group.append("rect")
    .attr("width", edge_width)
    .attr("height", "100%")
    .attr("fill", edge_color);
border_group.append("rect")
    .attr("x", "calc(100% - " + edge_width + "px)")
    .attr("width", edge_width)
    .attr("height", "100%")
    .attr("fill", edge_color);
border_group.append("rect")
    .attr("width", "100%")
    .attr("height", edge_width)
    .attr("fill", edge_color);
border_group.append("rect")
    .attr("y", "calc(100% - " + edge_width + "px)")
    .attr("width", "100%")
    .attr("height", edge_width)
    .attr("fill", edge_color);

// Load shapes
d3.tsv("shapes.tsv", shape_type, function(shape_error, shape_data) {
    // Load robots
    d3.tsv("robots.tsv", robot_type, function(robot_error, robot_data) {
        // SHAPES
        var shape_group = chart.append("g");
        var shapes = shape_group.selectAll("rect")
            .data(shape_data)
            .enter().append("rect");
        shapes
            .attr("x", function(d) { return d.x; })
            .attr("y", function(d) { return d.y; })
            .attr("width", function(d) { return d.width; })
            .attr("height", function(d) { return d.height; })
            .attr("fill", function(d) { return d3.rgb(d.r*c_dim, d.g*c_dim, d.b*c_dim); });

        // ROBOTS
        var robot_group = chart.append("g");
        var robots = robot_group.selectAll("circle")
            .data(robot_data)
            .enter().append("g");
            //.attr("transform", "translate(" + d.x + "," + d.y + ")");
        robots.append("path")
            .attr("d", d3.svg.symbol()
                .type(function(d) { return d.shape; })
                .size(55*55))
            .attr("transform", function(d) {
                var trans_str = "translate(" + d.x + "," + d.y + ")";
                var rot_str = "rotate(" + (d.theta*180/Math.PI + 90) + ")";
                return trans_str + rot_str;
            })
            .attr("fill", function(d) { return d3.rgb(d.r, d.g, d.b); });
        // Bearing
        robots.append("path")
            .attr("d", function(d) {
                var points = [[d.x, d.y],
                    [d.x+Math.cos(d.theta)*radius, d.y+Math.sin(d.theta)*radius]];
                var line_generator = d3.svg.line();
                return line_generator(points);
            })
            .attr("stroke", "black")
            .attr("stroke-width", 8)
            .attr("fill", "none");
    });
});

function shape_type(d) {
    d.x = +d.x;
    d.y = +d.y;
    d.width = +d.width;
    d.height = +d.height;
    d.r = +d.r;
    d.g = +d.g;
    d.b = +d.b;
    return d
}

function robot_type(d) {
    d.id = +d.id;
    d.x = +d.x;
    d.y = +d.y;
    d.theta = +d.theta;
    d.r = +d.r;
    d.g = +d.g;
    d.b = +d.b;
    return d
}