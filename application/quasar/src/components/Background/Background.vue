<template>
<canvas></canvas>
</template>

<script>
import _ from 'underscore';
import ParticleBackground from './Particle-Background';

export default {
    name: 'Background',

    data()  {
        return {
            background: null,
        }
    },

    methods: {

    },

    beforeCreate()  {},
	created()	{},
    beforeMount()   {},
	mounted() 	{
        let background = new ParticleBackground(this.$el);

        background.start(60, window.innerWidth, window.innerHeight);

        this.background = background;
    },
    beforeUpdate()  {},
    updated()   {},
    beforeDestroy() {},
    destroyed() {
        clearInterval(this.background._timer);
    }
}
</script>

<style scoped lang="less">
canvas {
    position: fixed;
    top:0;
    left:0;
    z-index:0;
}
</style>

