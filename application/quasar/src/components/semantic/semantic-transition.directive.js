import $ from 'jquery';


export default {
    bind: function(el, binding) {

    },

    inserted: function(el, binding)    {
        $(el).transition(binding.value);
    },

    update: function() {

    },

    componentUpdated: function() {

    },

    unbind: function(el) {

    },
};
