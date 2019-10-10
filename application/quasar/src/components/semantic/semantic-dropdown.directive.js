import $ from 'jquery';


class Dropdown {
    constructor(el, options = {
        on: 'click',
        allowReselection: false,
        allowAdditions: false,
        hideAdditions: true,
        action: 'activate',
        minCharacters: 1,
        match: 'both',
        transition: 'fade',
        selectOnKeydown: true,
        forceSelection: true,
        allowCategorySelection: false,
        placeholder: 'auto',
        apiSettings: false,
        fields: {
            remoteValues : 'results',
            values       : 'values',
            name         : 'name',
            value        : 'value'
        },
        filterRemoteData: false,
        saveRemoteData: true,
        useLabels: true,
        maxSelections: false,
        glyphWidth: 1.0714,
        label: {
            transition : 'horizontal flip',
            duration   : 200,
            variation  : false
        },
        direction: 'auto',
        keepOnScreen: true,
        context: window,
        fullTextSearch: false,
        preserveHTML: true,
        sortSelect: false,
        showOnFocus: true,
        allowTab: true,
        duration: 200,
        keys: {
            backspace  : 8,
            delimiter  : 188, // comma
            deleteKey  : 46,
            enter      : 13,
            escape     : 27,
            pageUp     : 33,
            pageDown   : 34,
            leftArrow  : 37,
            upArrow    : 38,
            rightArrow : 39,
            downArrow  : 40
        },
        delay: {
            hide   : 300,
            show   : 200,
            search : 50,
            touch  : 50
        },

        onChange: function()    {},
        onAdd: function()   {},
        onRemove: function()    {},
        onLabelCreate: function()   {},
        onLabelRemove: function()   {},
        onLabelSelect: function()   {},
        onNoResults: function() {},
        onShow: function()  {},
        onHide: function()  {},
    })   {
        this.$el = $(el).dropdown(options);
    };

    setupMenu() { this._behavior('setup menu'); };
    refresh() { this._behavior('refresh'); };
    toggle() { this._behavior('toggle'); };
    show() { this._behavior('show'); };
    hide() { this._behavior('hide'); };
    clear() { this._behavior('clear'); };
    hideOthers() { this._behavior('hide others'); };
    restoreDefaults() { this._behavior('restore defaults'); };
    restoreDefaultText() { this._behavior('restore default text'); };
    restorePlaceholderText() { this._behavior('restore placeholder text'); };
    restoreDefaultValue() { this._behavior('restore default value'); };
    saveDefaults() { this._behavior('save defaults'); };
    removeSelected(value) { this._behavior('remove selected', value); };
    setSelected(value) { this._behavior('set selected', value); };
    setExactly(values) { this._behavior('set exactly', values[0], values[1]); };
    setText(text) { this._behavior('set text', text); };
    setValue(value) { this._behavior('set value', value); };
    getText() { return this._behavior('get text'); };
    getValue() { return this._behavior('get value'); };
    getItem(value) { return this._behavior('get item', value); };
    bindTouchEvents() { this._behavior('bind touch events'); };
    bindMouseEvents() { this._behavior('bind mouse events'); };
    bindIntent() { this._behavior('bind intent'); };
    unbindIntent() { this._behavior('unbind intent'); };
    determineIntent() { return this._behavior('determine intent'); };
    determineSelectAction(text, value) { this._behavior('determine select action', text, value); };
    setActive() { this._behavior('set active'); };
    setVisible() { this._behavior('set visible'); };
    removeActive() { this._behavior('remove active'); };
    removeVisible() { this._behavior('remove visible'); };
    isSelection() { return this._behavior('is selection'); };
    isAnimated() { return this._behavior('is animated'); };
    isVisible() { return this._behavior('is visible'); };
    isHidden() { return this._behavior('is hidden'); };
    getDefaultText() { return this._behavior('get default text'); };
    getPlaceholderText() { return this._behavior('get placeholder text'); };

    _behavior(...args) {
        return $.fn.dropdown.apply(this.$el, args);
    };
};

export default {
    bind: function(el, binding) {

    },

    inserted: function(el, binding, vnode)    {
        var dropdown = new Dropdown(el, binding.value.setting);

        vnode.context[binding.value.el] = dropdown;
    },

    update: function() {

    },

    componentUpdated: function() {

    },

    unbind: function(el) {

    },
};
