class Point {
    constructor(x = 0, y = 0)   {
        this.x = x;
        this.y = y;
    };

    static getDistance(point1 = new Point, point2 = new Point)  {
        return Math.sqrt(Math.pow((point2.x - point1.x), 2) + Math.pow((point2.y - point1.y), 2));
    };
};

class Linear    {
    constructor()   {
    };
};

