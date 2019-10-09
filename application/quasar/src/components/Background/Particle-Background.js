import _ from 'underscore';
import Particle  from './Particle';


class Background    {
    constructor($el = HTMLCanvasElement.prototype)   {
        this.canvas = $el;
        this.ctx = $el.getContext('2d');
        this.particles = [];
        this._timer = -1;

        $el.addEventListener('resize', this.initCanvasSize);
    };

    initCanvasSize(width, height) {
        this.canvas.width = width || window.innerWidth;
        this.canvas.height = height || window.innerHeight;

        Particle.MAX_X = this.canvas.width;
        Particle.MAX_Y = this.canvas.height;
    };

    createParticle(maxParticleCount) {
        for (let i = 0; i < maxParticleCount; i++) {
            let particle = new Particle({});

            this.particles.push(particle);
            particle.create();
            particle.color = {
                r: _.random(0, 255), g: _.random(0, 255), b: _.random(0, 255)
            };

            particle.autoMove();
        }
    };

    drawParticles() {
        let ctx = this.ctx;

        this.particles.forEach(particle => {
            ctx.save();
            ctx.beginPath();

            ctx.fillStyle = `rgba(${particle.color.r}, ${particle.color.g}, ${particle.color.b}, ${particle.opacity})`;

            ctx.arc(particle.x, particle.y, 2.5, 0, Math.PI * 2);
            ctx.fill();

            ctx.closePath();
            ctx.restore();
        });
    };

    drawParticleConnectionLine() {
        let ctx = this.ctx;

        this.particles.forEach((p1, a) => {
            for (let i = a + 1; i < this.particles.length; i++) {
                let p2 = this.particles[i],
                    sx = Math.pow(p2.x - p1.x, 2),
                    sy = Math.pow(p2.y - p1.y, 2),
                    s = Math.sqrt(sx + sy),
                    opacity = 5 / s;

                opacity <= 0.028 ? opacity = 0 : opacity;

                if (opacity > 0) {
                    ctx.save();
                    ctx.beginPath();

                    ctx.strokeStyle = `rgba(0, 0, 0, ${opacity})`;

                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();

                    ctx.closePath();
                    ctx.restore();
                }
            }
        });
    };

    start(maxParticleCount, width, height) {
        this._timer = setInterval(() => {
            this.initCanvasSize(width, height);
            this.drawParticles();
            this.drawParticleConnectionLine();
        }, 50);

        setTimeout(() => {
            this.createParticle(maxParticleCount);
        }, 51);
    };
};

export default Background;
