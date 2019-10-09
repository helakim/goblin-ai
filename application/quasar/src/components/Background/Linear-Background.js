import _ from 'underscore';

class Background    {
    constructor($el = HTMLCanvasElement.prototype)   {
        this.canvas = $el;
        this.ctx = $el.getContext('2d');
        this._timer = -1;

        // $el.addEventListener('resize', this.initCanvasSize);
    };

    initCanvasSize(width, height) {
        this.canvas.width = width || window.innerWidth;
        this.canvas.height = height || window.innerHeight;
    };

    createLinear(direction = 0, point = { x: 0, y: 0 })  {

    };

    start(width, height) {
        this._timer = setInterval(() => {
            this.initCanvasSize(width, height);
        }, 50);
    };
};

const LIMIT_LENGTH = 50;

const NORTH = 1;
const EAST = 2;
const SOUTH = 3;
const WEST = 4;
const NORTH_WEST = 5;
const NORTH_EAST = 6;
const BOTTOM_EAST = 7;
const BOTTOM_WEST = 8;

const DIRECTION = {
    NORTH: NORTH,
    EAST: EAST,
    SOUTH: SOUTH,
    WEST: WEST,
    NORTH_WEST: NORTH_WEST,
    NORTH_EAST: NORTH_EAST,
    BOTTOM_EAST: BOTTOM_EAST,
    BOTTOM_WEST: BOTTOM_WEST,
};
