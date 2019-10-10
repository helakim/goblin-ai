import $ from 'jquery';


class Modal {
    constructor(el = document.querySelector('.ui.modal'), options = {
        detachable: true,
        autofocus: true,
        observeChanges: false,
        allowMultiple: false,
        keyboardShortcuts: true,
        offset: 0,
        context: 'body',
        closable: true,
        dimmerSettings: { closable: false, useCSS: true },
        transition: 'scale',
        duration: 400,
        queue: false,
        inverted: false,

        onShow: function()  {},
        onVisible: function()   {},
        onHide: function($) {},
        onHidden: function()    {},
        onApprove: function($)  {},
        onDeny: function($) {},
    })   {
        // if(el.parentElement.constructor !== HTMLBodyElement)    {
        //     document.querySelector('body').insertBefore(el, null);
        // }

        this.$el = $(el).modal(options);
    };

    show()      { return this._behavior('show'); };
    hide()      { return this._behavior('hide'); };
    toggle()    { return this._behavior('toggle'); };
    refresh()    { return this._behavior('refresh'); };
    showDimmer()    { return this._behavior('show dimmer'); };
    hideDimmer()    { return this._behavior('hide dimmer'); };
    hideOthers()    { return this._behavior('hide others'); };
    hideAll()    { return this._behavior('hide all'); };
    cacheSizes()    { return this._behavior('cache sizes'); };
    canFit()    { return this._behavior('can fit'); };
    remove()    { this.$el.remove(); };

    _behavior(...args) {
        return $.fn.modal.apply(this.$el, args);
    };
};



export default {
    bind: function(el, binding) {

    },

    inserted: function(el, binding, vnode)    {
        var modal = new Modal(el, binding.value.setting);

        vnode.context[binding.value.el] = modal;
    },

    update: function() {

    },

    componentUpdated: function() {

    },

    unbind: function(el) {
        el.remove();
    },
};
