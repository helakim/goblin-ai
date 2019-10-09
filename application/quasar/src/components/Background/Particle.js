import _ from 'underscore';

class Particle  {
    constructor({
        x = 0,
        y = 0,
        vx = 0,
        vy = 0,
        opacity = Math.random(),
        color = {
            r: 0,
            g: 0,
            b: 0
        },
        vOpacity = .02,
    })   {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.opacity = opacity;
        this.color = color;
        this.vOpacity = vOpacity;

        setInterval(() => {
            if(this.opacity >= 1)	this.vOpacity *= -1;
            else if(this.opacity <= 0.02)	this.vOpacity *= -1;

            this.opacity += this.vOpacity;
        }, 50);
    };

    create()    {
        this.x = _.random(0, Particle.MAX_X);
        this.y = _.random(0, Particle.MAX_Y);

        this.vx = Math.random() * 1 * (_.random(0, 4) % 2 == 0 ? 1 : -1);
        this.vy = Math.random() * 1 * (_.random(0, 4) % 2 == 0 ? 1 : -1);
    };

    autoMove()  {
        setInterval(() => {
            this.x >= Particle.MAX_X || this.x <= 0 ? this.vx *= -1 : null;
            this.y >= Particle.MAX_Y || this.y <= 0 ? this.vy *= -1 : null;

            this.x += this.vx;
            this.y += this.vy;
        }, 50);
    };
};

Particle.MAX_X = 0;
Particle.MAX_Y = 0;

export default Particle;
