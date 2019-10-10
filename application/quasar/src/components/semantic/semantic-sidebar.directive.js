import $ from 'jquery';


class Sidebar {
    constructor(el, options = {
        context: document.querySelector('body'),
        exclusive: false,
        closable: true,
        dimPage: true,
        scrollLock: false,
        returnScroll: false,
        delaySetup: false,

        transition: 'auto',
        mobileTransition: 'auto',
        defaultTransition: {
            computer: {
                left   : 'uncover',
                right  : 'uncover',
                top    : 'overlay',
                bottom : 'overlay'
            },
            mobile: {
                left   : 'uncover',
                right  : 'uncover',
                top    : 'overlay',
                bottom : 'overlay'
            }
        },
        useLegacy: false,
        duration: 500,
        easing: 'easeIn',

        onVisible: function()   {},
        onShow: function()   {},
        onChange: function()   {},
        onHide: function()   {},
        onHidden: function()   {},
    })   {
        // if(el.parentElement.constructor !== HTMLBodyElement)    {
        //     var pusher = document.querySelector('.pusher');

        //     document.querySelector('body').insertBefore(el, pusher);
        // }

        this.$el = $(el).sidebar(options);
    };

    attachEvents(selector, event) { this._behavior('attach events', selector, event); };
    show() { this._behavior('show'); };
    hide() { this._behavior('hide'); };
    toggle() { this._behavior('toggle'); };
    isVisible() { return this._behavior('is visible'); };
    isHidden() { return this._behavior('is hidden'); };
    pushPage() { this._behavior('push page'); };
    getDirection() { return this._behavior('get direction'); };
    pullPage() { this._behavior('pull page'); };
    addBodyCSS() { this._behavior('add body CSS'); };
    removeBodyCSS() { this._behavior('remove body CSS'); };
    getTransitionEvent() { return this._behavior('get transition event'); };
    setAnimation(animation) { this._behavior('setting', 'transition', animation); };

    _behavior(...args) {
        return $.fn.sidebar.apply(this.$el, args);
    };
};

export default {
    bind: function(el, binding, vnode) {

    },

    inserted: function(el, binding, vnode, oldVnode)    {
        var sidebar = new Sidebar(el, binding.value.setting),
            animation = binding.value.animation;

        vnode.context[binding.value.el] = sidebar;

        switch(animation) {
            case 'push':
            case 'overlay':
                sidebar.setAnimation(animation);
                break;
            default:
                break;
        }
    },

    update: function() {

    },

    componentUpdated: function() {

    },

    unbind: function(el) {

    },
};
